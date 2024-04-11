import jax
import numpy as np
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.experimental import mesh_utils
from typing import Sequence
from absl import app
import os
import argparse
from typing import Optional
from layers import quantizations
from layers import simple_decoder_layer
from layers import llama2
import common_types
import pyconfig
import functools
import max_utils

def stack_pytrees(*pytrees):
  """Stacks pytrees with identical structure along a new leading dimension."""
  def stacking_fn(*leaves):
    return jnp.stack(leaves)
  return tree_map(stacking_fn, *pytrees)

def create_mesh(n_stages, tp_axis, dp_axis):
  devices = mesh_utils.create_device_mesh((n_stages, tp_axis, dp_axis))
  

  mesh = Mesh(devices, axis_names=('stage', 'tensor', 'data'))
  return mesh

def get_weights_and_inputs(batch_size, sequence, features, n_layers):
    '''Get random weights, random inputs, and random targets

        Returns
            weights: [n_layers, features, features]
            inputs: [global_batch, sequence, features]
            targets: [global_batch, sequence, features]
    '''
    weights_shape = jnp.array([n_layers, features, features]) # pytree in real cases instead of single array
    k = jax.random.PRNGKey(1)
    weights = jax.random.normal(k,weights_shape, dtype=jnp.float32)

    # we pass in input with global batch, its up to the pipeline function to reshape to microbatches
    input_shape = [batch_size, sequence, features]
    k = jax.random.PRNGKey(2)
    inputs = jax.random.normal(k,input_shape, dtype=jnp.float32)
    
    # dummy targets same shape as inputs to use for a dummy loss funciton to check gradient correctness
    k = jax.random.PRNGKey(3)
    dummy_targets = jax.random.normal(k,input_shape, dtype=jnp.float32)

    inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
    print(f"{inputs_position.shape}") 
    #inputs_position = jnp.arange((batch_size, sequence), dtype = jnp.int32)
    inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)

    return weights, inputs, dummy_targets, inputs_position, inputs_segmentation

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  # TODO: Some properties, (num repeat, num_micro are through the config, some are derived. This makes it annoying to call diff properties. Is there a better solution?)
  # TODO: Should we declare the derived properties here as well? I think anything declared here becomes required as an input
  config: common_types.Config
  decoder_layer_class: nn.Module
  mesh: common_types.Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    # TODO: See what Inputs are needed to initialize DecoderLayers e.g. LlamaDecoderLayer
    decoder_layers = [self.decoder_layer_class(config=self.config, mesh=self.mesh, name=f'layers_{lyr}', quant=self.quant) for lyr in range(self.config.num_decoder_layers)]
    self.decoder_layers = decoder_layers
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.layers_per_stage = self.config.num_decoder_layers / (self.num_stages * self.config.num_pipeline_repeats)
    # TODO: should this assert be in this class or in the initial pyconfig check?
    assert self.layers_per_stage==1,f"Currently only supporting 1 layer per pipeline stage, but {self.config.num_decoder_layers} layers were requested with {self.num_stages} stages"
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    # TODO: improve error message to show inputs
    assert microbatches_per_stage * self.num_stages == self.config.num_pipeline_microbatches, f"Currently the number of microbatches ({self.config.num_pipeline_microbatches}) must be divisible by the number of stages ({self.num_stages})"
    self.microbatches_per_stage = microbatches_per_stage

    
  def S(self, *specs):
    return NamedSharding(self.mesh, PartitionSpec(*specs))

  def shard_dim_by_stages(self, x):
   '''Assumes the stages dimension is leading and the mesh has name stages.'''
   # TODO: currently uses physical axes instead of logical, should we use logical instead?
   specs = ['pipeline_stage'] + [None] * (x.ndim - 1)
   stage_sharding = self.S(*specs)
   return jax.lax.with_sharding_constraint(x, stage_sharding)
  
  def init_states(self, inputs):
    '''Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns
          shift: zeros shape [num_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
          circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed]
          circ_storage_mover: zeros[num_stages, micro_size, sequence, embed]
    
    '''

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    #shift = self.shard_dim_by_stages(shift, self.mesh)
    # TODO: Is there a standard way to go from logical -> physical instead of the logical_to_mesh followed by with_sharding_constraint?
    # Answer: yes probably nn.with_logical_constraint https://github.com/google/maxtext/blob/5deb01a9221612c6c28f3f5a561b8af9c0fd720d/MaxText/layers/models.py#L322
    # Can remove mesh and logical_axis_rules and rely on global context manager (but we should remove the global context manager anyway at some point) 
    shift_shardings = nn.logical_to_mesh_axes(["activation_stage", "activation_batch", "activation_length", "activation_embed"],self.config.logical_axis_rules) 
    shift = jax.lax.with_sharding_constraint(shift,NamedSharding(self.mesh, shift_shardings))
    #shift = jax.lax.with_sharding_constraint(shift, self.S(self.mesh, *shift_shardings)) # For some reason this complains about resource names
    #shift = jax.lax.with_sharding_constraint(shift, S(self.mesh, 'stage', 'data', None, 'tensor'))

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    #state_io = self.shard_dim_by_stages(state_io)
    state_io_shardings = nn.logical_to_mesh_axes(["activation_stage", None, "activation_batch", "activation_length", "activation_embed"],self.config.logical_axis_rules) 
    state_io = jax.lax.with_sharding_constraint(state_io, NamedSharding(self.mesh, state_io_shardings))
    #state_io = jax.lax.with_sharding_constraint(state_io, self.S(self.mesh, *state_io_shardings))

    # TODO: verify comment below
    # The data/fsdp can shard over microbatch_size, not number of microbatches. The num_microbatches is looped over so should not be sharded.

    # TODO: Consider sharding and/or changing the circ storage
    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage can immediately pass to the first without additional storage.
    # Alternative name is "between_repeats_storage"
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed] -- this is huge btw, it should be reducible by a factor of num_stages
    if self.use_circ_storage:
        circ_storage = jnp.zeros((self.num_stages,) + inputs.shape , dtype=inputs.dtype)
    else:
       circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage
    # circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
    # This mover is one iteration behind before being pushed into storage - which is why we can't just re-use output
    # However shouldn't we be able to keep only the last stage's output instead of all stages?
    if self.use_circ_storage:
        circ_storage_mover = shift
    else:
       circ_storage_mover = None

    return state_io, shift, circ_storage, circ_storage_mover

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    '''
    Construct stages_in: the global array that is operated on for this iteration, shape same as shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab from state_io or circ_storage
    '''

    # Setup potential input from state_io. state_io has a rotating microbatch index (size of micro/stages, stream_buf_idx below)
    state_io_batch_idx = loop_iteration % (self.config.num_pipeline_microbatches // self.num_stages)
    state_io_slice = state_io[:,state_io_batch_idx] 

    if self.use_circ_storage:
        # Setup potential input from circ_slice, which also has a rotating index for microbatch
        circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
        circ_storage_slice = circ_storage[:,circ_storage_batch_idx]
    else:
        circ_storage_slice = shift

    stages_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circ_storage_slice)

    def select_state_or_input(input, shift):
        # Selects input for stage 0, shift for other stages
        return jnp.where(jax.lax.broadcasted_iota('int32', shift.shape, 0) == 0, input, shift)

    # Selects input (from stream_io or circ_slice) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(stages_in, shift)
    return stages_in

  def get_weights_stage(self, weights, loop_iteration):
    '''
    Get the weights for each stage used for this loop itereation. 
    
    Input:
        Weights are a pytree where each leaf has a leading dimension of num_layers, e.g. [num_layers, embed, mlp]
    Returns:
        Weights of same pytree structure but each leaf has a leading dimension of num_stages, e.g. [num_stages, embed, mlp].

    For non-circular pipelines this would just be stacked [weights_layer_0; weights_layer1; etc],
    but for circular the stages need a repeat_idx to determine what layer weights to grab, e.g. on iteration 5 with 4 stages
    the repeat indexes are [1,1,0,0] so need layers [4,5,2,3]
    '''
    # We use numpy instead of jnp so these indexes are not traced
    microbatch_ids = np.maximum(loop_iteration - np.arange(self.num_stages), 0) # not a great name, this is really batch_id * repeat idx
    repeat_ids = microbatch_ids // self.config.num_pipeline_microbatches
    layer_ids = np.arange(self.num_stages) + repeat_ids * self.num_stages
    #layer_ids goes out of bounds on the last bubble, we cap it within range.
    layer_ids= np.minimum(layer_ids, self.config.num_decoder_layers - 1)
    # slice_in_dim avoids executing an all gather


    def layers_dimension_to_stages(weight_leaf):
       weights_stage_list= [jax.lax.slice_in_dim(weight_leaf,layer_ids[stage], layer_ids[stage] + 1, axis=0) for stage in range(self.num_stages)]
       weights_stage = jnp.concatenate(weights_stage_list, axis=0)
       weights_stage_shape = (self.num_stages,) + weight_leaf.shape[1:]
       weights_stage = jnp.reshape(weights_stage, weights_stage_shape)
       return weights_stage # This reshape unsqueezes singleton axes that were potentially squeezed in concatenate
    weights_stage = jax.tree_map(layers_dimension_to_stages, weights)
    return weights_stage

  # TODO: should we pass in the weights explicitly? How about the segmentation IDs

  def get_microbatch_id(self, stage_idx, loop_iteration):
    '''
    Gets the microbatch_id on this loop_iteration for this stage.
    
    Input:
        stage_idx: Index of this stage, integer from 0 to num_stages - 1
        loop_iteration: Integer of loop index
    Returns:
        Integer representing which microbatch the stage at stage_idx will work on during loop_iteration
    '''
    return (loop_iteration - stage_idx) % self.config.num_pipeline_microbatches
     
  def get_microbatches_for_stages(self, microbatched_array, loop_iteration):
    '''
    Returns an array of leading dimension stages grabbing the current microbatch for each stage.
    TODO: This is not actually used to get the microbatches, but the position/segment IDs, so probably should change method name
    
    Input:
        microbatched_array: Array to grab from, should have leading dimension num_microbatches
        loop_iteration: Integer of loop index
    Returns:
        Array of shape microbatched_array, except the leading dimension is replaced by num_stages
    '''

    microbatched_stages_list = [microbatched_array[self.get_microbatch_id(stage_idx, loop_iteration)] for stage_idx in range(self.num_stages)]
    stages_array = jnp.concatenate(microbatched_stages_list, axis=0)
    stages_array = jnp.reshape(stages_array, (self.num_stages,) + microbatched_array.shape[1:])
    return stages_array

  def get_new_loop_state(self,output, old_state_io, old_circ_storage, old_circ_storage_mover, loop_iteration):
    '''
      Update the various buffers given the output of the most recent iteration
      * state_io: rotates left/up by 1 (replace last element with last stage output) - we are pushing inputs up into the pipeline
      * shift: rotate output right/down by 1 - we imagine the pipeline moves to right/down
      * circ_storage: push latest circ_mover (e.g. FULL outputs) into rotating index -- why are we pushing full ouputs, why not just last stage?
      * circ_mover gets FULL? rotated output -- I think it should only need the last stage of output
    '''

    # Shift becomes a rotated-right version of the previous output
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)
    #breakpoint()
    jit_rotate_right = jax.jit(_rotate_right)
    new_shift = jit_rotate_right(output)
    #new_shift = _rotate_right(output) #TODO(big):file a bug or ping again on jax chat, why do we need to jit here

    if self.use_circ_storage:
        # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
        # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped compute/async transfers
        def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
            rotated = _rotate_right(circ_storage_mover_in)
            rotated = jnp.expand_dims(rotated, 1)
            # The offset is the last stage's last microbatch ID. 
            offset = (loop_iteration - (self.num_stages - 1) - 1) % self.num_pipeline_microbatches # Note extra -1 b/c grabbing from the previous output - circ_storage_mover is one iter behind
            return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)
        new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
        new_circ_storage_mover = output
    else:
       new_circ_storage = None
       new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating ms index (stream_buf_idx), replacing the last/bottom with the last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]
    def _update_state_io(state_in, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == self.num_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state_in, stream_slice, stream_buf_idx, axis=1)
    jit_update_state_io = jax.jit(_update_state_io)
    new_state = jit_update_state_io(old_state_io, stream_slice, output) # TODO(medium):same bug, requires jit
    #new_state = _update_state_io(old_state_io, stream_slice, output)

    return new_state, new_shift, new_circ_storage, new_circ_storage_mover
   
  def permute_output_ms_dim(self, output):
    '''
    Although re-using the same array for both input and output is cute,
    The final outputs turn out permuted compared to the inputs. Worringly I don't see this function in praxis
    '''

    # The first real output (batch 0) takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on the number of iters
    first_output_num_iters = self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + self.num_stages - 1
    # The first term above is a multiple of num_pipeline_microbatches and thus could be ignored since its also a multiple of microbatches_per_stage
    land_idx = first_output_num_iters % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + land_idx) % self.microbatches_per_stage # make the value in land_idx actually appear in idx 0, and (land_idx + 1) appear in spot 1, etc
    output = output[:,permutation]
    return output

  def run_one_iteration(self, state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, positions, segment_ids, deterministic, model_mode):
   '''
      Run one loop iteration - sending inputs and specifying weights for each pipeline stage, run the pipeline, and update the various state buffers
   '''
   stages_weights = self.get_weights_stage(weights, loop_iteration)
   stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
   if positions is not None:
    stages_positions = self.get_microbatches_for_stages(positions, loop_iteration)
    positions_stage_idx = 0
   else:
     stages_positions = None
     positions_stage_idx = 0
   if segment_ids is not None:
    stages_segment_ids = self.get_microbatches_for_stages(segment_ids, loop_iteration)
    segment_stage_idx = 0
   else:
    stages_segment_ids
    segment_stage_idx = None
   stages_output = jax.vmap(self.decoder_layers[0].apply, in_axes=[0,0,positions_stage_idx, segment_stage_idx, None, None])(stages_weights, stages_inputs, stages_positions, stages_segment_ids, deterministic, model_mode)
   new_state_io, new_shift, new_circ_storage, new_circ_storage_mover = self.get_new_loop_state(stages_output, state_io, circ_storage, circ_storage_mover, loop_iteration)
   return new_state_io, new_shift, new_circ_storage, new_circ_storage_mover
  
  def __call__(self, inputs: jnp.ndarray, positions: jnp.ndarray, segment_ids:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    # We want to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict)
    # TODO: may want to have some simplified flow when is initializing instead (don't need to run through total_iters)
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    if positions is not None:
      positions = positions.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      positions_0 = positions[0]
    else:
      positions_0 = None
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      segment_ids_0 = segment_ids[0]
    else:
      segment_ids_0 = None
    for decoder in self.decoder_layers:
      # Initialize the decoder variables, since they are lazily initialized and we need them now.
      _ = decoder(inputs[0], positions_0, segment_ids_0, deterministic, model_mode)
         


    state_io, shift, circ_storage, circ_storage_mover = self.init_states(inputs)
    total_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats + self.num_stages  - 1 
    # TODO(huge): Shard the weights. This may be tricky b/c there is no "stage" axis in the weights to shard over until after the below
    weights = [decoder.variables for decoder in self.decoder_layers]
    # Go from a list of size n_layers of weight pytrees to a single pytree where each leaf has a leading dimension of n_layers 
    weights = stack_pytrees(*weights)
    for loop_iteration in range(total_iterations):
       print(f"starting iteration {loop_iteration}")
       state_io, shift, circ_storage, circ_storage_mover = self.run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, positions, segment_ids, deterministic, model_mode)

    # The final output is located in the input/output array, however the microbatches may be permuted
    final_output = self.permute_output_ms_dim(state_io)

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    # TODO: Either confirm or fix batch size when mixed sharding (e.g. is this correct even with PP + FSDP?)
    final_output = jnp.reshape(final_output, (self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
                               
    return final_output

def main(argv: Sequence[str]) -> None:
  # This only exists for convenient testing

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config

  # TODO: determine if num_stages should be added to pyconfig or elsewhere
  num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
  layers_per_stage = config.num_decoder_layers / (num_stages * config.num_pipeline_repeats)
  #assert layers_per_stage==1,"Currently only supporting 1 layer per pipeline stage"

  _, inputs, targets, inputs_position, inputs_segmentation = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)
  deterministic = False
  model_mode = common_types.MODEL_MODE_TRAIN

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  #mesh = create_mesh(num_stages, config.ici_tensor_parallelism, config.ici_data_parallelism)

  
  #block_layer = simple_decoder_layer.SimpleDecoderLayer
  block_layer = llama2.LlamaDecoderLayer


  if 1:
    my_pipeline = Pipeline(
      config=config,
      decoder_layer_class=block_layer,
      mesh=mesh
    )
    init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
    my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
  else:
    llama_layer = block_layer(config=config,mesh=mesh, name=f'layers_{0}') 
    inputs=inputs[0:2]
    inputs_position=inputs_position[0:2]
    inputs_segmentation=inputs_segmentation[0:2]
    init_llama = llama_layer.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
    print("HELLO WORLD!", flush=True)
    inputs=inputs[0:2]
    inputs_position=inputs_position[0:2]
    inputs_segmentation=inputs_segmentation[0:2]
    print(inputs.shape)
    llama_layer.apply(init_llama, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)


if __name__ == "__main__":
  app.run(main)