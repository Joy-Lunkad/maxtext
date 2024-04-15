# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Pipeline layer wrapping a decoder layer. Supports circular pipelining '''

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
import common_types
import pyconfig
import functools
import max_utils
print("\n\n\n HELLO WORLD 329823 \n\n\n")

def stack_pytrees(*pytrees):
  """Returns a pytree with identical key structure but whose leaves have a new [num pytree inputs] leading dimension stacking from each input."""
  def stacking_fn(*leaves):
    return jnp.stack(leaves)
  return tree_map(stacking_fn, *pytrees)

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  # TODO: Some properties, (num repeat, num_micro are through the config, some are derived. This makes it annoying to call different properties, some are self.property, some are self.config.property)
  # TODO: Should we declare the derived properties here as well (e.g. num_stages? I think anything declared here becomes required as an input though
  config: common_types.Config
  decoder_layer_class: nn.Module
  mesh: common_types.Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    print("\n\n\n HELLO WORLD 27 \n\n\n")
    #decoder_layers = [self.decoder_layer_class(config=self.config, mesh=self.mesh, name=f'layers_{lyr}', quant=self.quant) for lyr in range(self.config.num_decoder_layers)]
    #self.decoder_layers = decoder_layers
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.layers_per_stage = self.config.num_decoder_layers / (self.num_stages * self.config.num_pipeline_repeats)
    # TODO: should this assert be in this class or in the initial pyconfig check?
    assert self.layers_per_stage==1,f"Currently only supporting 1 layer per pipeline stage, but {self.config.num_decoder_layers} layers were requested with {self.num_stages} stages"
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    # TODO: should this assert be in this class or pyconfig check?
    assert microbatches_per_stage * self.num_stages == self.config.num_pipeline_microbatches, f"Currently the number of microbatches ({self.config.num_pipeline_microbatches}) must be divisible by the number of stages ({self.num_stages})"
    self.microbatches_per_stage = microbatches_per_stage

    
  def S(self, *specs):
    return NamedSharding(self.mesh, PartitionSpec(*specs))

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
    shift = nn.with_logical_constraint(shift, ("activation_stage", "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules,mesh=self.mesh)

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = nn.with_logical_constraint(state_io, ("activation_stage", None, "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules, mesh=self.mesh) 

    # TODO: verify comment below
    # The data/fsdp can shard over microbatch_size, not number of microbatches. The num_microbatches is looped over so should not be sharded.

    # TODO: Consider sharding and/or changing the circ storage. Understand if/when setting microbatches > num_stages is beneficial which is when circ_storage is needed
    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage can immediately pass to the first without additional storage.
    # Alternative name is "between_repeats_storage", since this storage is not needed always for circular - but only when microbatches > num_stages
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
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from either state_io or circ_storage
    '''

    # Setup potential input from state_io, which has a rotating microbatch index (size of micro/stages, state_io_batch_idx below)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:,state_io_batch_idx] 

    if self.use_circ_storage:
        # Setup potential input from circ_storage, which also has a rotating index for microbatch, size of num_microbatches
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
        Weights are a pytree where each leaf has a leading dimension of num_layers, example leaf shape: [num_layers, embed, mlp]
    Returns:
        Weights of same pytree structure but each leaf has a leading dimension of num_stages, example leaf shape: [num_stages, embed, mlp].

    For non-circular pipelines this would just be stacked [weights_layer_0; weights_layer1; etc],
    but for circular the stages need a repeat_idx to determine what layer weights to grab, e.g. on iteration 5 with 4 stages
    the repeat indexes are [1,1,0,0] so need layers [4,5,2,3]
    '''
    # TODO(huge): When the weights are correctly sharded on init (not implemented currently) we need to ensure that this
    # function does not execute an all-gather: All of the layers weights can be initialized on the correct devices and shouldn't
    # need to be communicated to others (at least for forward pass, need some thought for backward whether this is necessary)

    # We use numpy instead of jnp so these indexes are not traced
    microbatch_ids = np.maximum(loop_iteration - np.arange(self.num_stages), 0) # not a great name, this is really something like microbatch_id * repeat idx
    repeat_ids = microbatch_ids // self.config.num_pipeline_microbatches
    layer_ids = np.arange(self.num_stages) + repeat_ids * self.num_stages
    #layer_ids goes out of bounds on the last bubble, we cap it within range.
    layer_ids= np.minimum(layer_ids, self.config.num_decoder_layers - 1)
    
    def layers_dimension_to_stages(weight_leaf):
       # slice_in_dim avoids executing an all gather
       weights_stage_list= [jax.lax.slice_in_dim(weight_leaf,layer_ids[stage], layer_ids[stage] + 1, axis=0) for stage in range(self.num_stages)]
       weights_stage = jnp.concatenate(weights_stage_list, axis=0)
       weights_stage_shape = (self.num_stages,) + weight_leaf.shape[1:]
       weights_stage = jnp.reshape(weights_stage, weights_stage_shape) # This reshape unsqueezes singleton axes that were potentially squeezed in concatenate
       return weights_stage
    weights_stage = jax.tree_map(layers_dimension_to_stages, weights)
    return weights_stage

  def get_microbatch_id(self, stage_idx, loop_iteration):
    '''Gets the microbatch_id on this loop_iteration for this stage. Works for both circular and non-circular'''
    return (loop_iteration - stage_idx) % self.config.num_pipeline_microbatches
     
  def get_microbatches_for_stages(self, microbatched_array, loop_iteration):
    '''
    Returns an array of leading dimension stages grabbing the current microbatch for each stage.
    TODO: This is not actually used to get the microbatches, but the position and segment IDs, so probably should change method name
    
    Input:
        microbatched_array: Array to grab from, should have leading dimension num_microbatches
        loop_iteration: Integer of loop index
    Returns:
        One microbatch from microbatched_array for each stage. Array same shape as microbatched_array except the leading dimension is replaced by num_stages
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
    #new_shift = _rotate_right(output) #TODO(big):file a bug or ping again on jax chat, why do we need to jit here
    jit_rotate_right = jax.jit(_rotate_right)
    new_shift = jit_rotate_right(output)

    if self.use_circ_storage:
        # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
        # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped compute/async transfers
        def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
            rotated = _rotate_right(circ_storage_mover_in)
            rotated = jnp.expand_dims(rotated, 1)
            # The offset is the last stage's last microbatch ID. 
            offset = (loop_iteration - (self.num_stages - 1) - 1) % self.config.num_pipeline_microbatches # Note extra -1 b/c grabbing from the previous output - circ_storage_mover is one iter behind
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
    #new_state = _update_state_io(old_state_io, stream_slice, output)# TODO(medium):same sharding/jit issue
    jit_update_state_io = jax.jit(_update_state_io)
    new_state = jit_update_state_io(old_state_io, stream_slice, output) 
    
    return new_state, new_shift, new_circ_storage, new_circ_storage_mover
   
  def permute_output_ms_dim(self, output):
    '''
    TODO: Reach out to praxis owner about this permutation, fix comment and method name
    Although re-using the same array for both input and output is cute,
    The final outputs turn out permuted compared to the inputs. Worringly I don't see this function in praxis
    '''

    # The first real output (batch 0) takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on the number of iters
    first_output_num_iters = self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + self.num_stages - 1
    # The first term above is a multiple of num_pipeline_microbatches and thus could be ignored since its also a multiple of microbatches_per_stage, but we keep it for clairty
    land_idx = first_output_num_iters % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + land_idx) % self.microbatches_per_stage # make the value in land_idx actually appear in idx 0, and (land_idx + 1) appear in spot 1, etc
    output = output[:,permutation]
    return output

  def run_one_iteration(self, state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, positions, segment_ids, deterministic, model_mode):
   '''Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel, and update the various state buffers'''
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
  
  @nn.compact
  def __call__(self, inputs: jnp.ndarray, positions: jnp.ndarray, segment_ids:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    # Reshape inputs of [global_batch, ...] to [microbatches, microbatch_sizes, ...]
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    if positions is not None:
      positions = positions.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      example_position = positions[0]
    else:
      example_position = None
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      example_segmentation = segment_ids[0]
    else:
      example_segmentation = None

    loop_iteration = 0
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

    if self.is_initializing():
      sdl = simple_decoder_layer.SimpleDecoderLayer
      vmapped_fn = nn.vmap(
        sdl,
        in_axes=(0,positions_stage_idx, segment_stage_idx, None, None),
        out_axes = 0,
        spmd_axis_name="stage",
        variable_axes={'params': 0},
        split_rngs={'params': True},
        metadata_params={"partition_name": "layers"}
      )
      instantiated_vmapped_fn = vmapped_fn(config=self.config, mesh=self.mesh, name='layers')
      def reshape_to_layers(arr):
        return jnp.broadcast_to(arr[0], (self.config.num_decoder_layers,) + arr.shape[1:])

      input_init = reshape_to_layers(inputs)
      positions_init = reshape_to_layers(positions)
      segments_init = reshape_to_layers(segment_ids)
      return instantiated_vmapped_fn(input_init, positions_init, segments_init, deterministic, model_mode)
    # We need to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict since lazy initialization)
    
    weights = self.variables
    breakpoint()
    for decoder in self.decoder_layers:
      # Initialize the decoder variables, since they are lazily initialized and we need them now.
      _ = decoder(inputs[0], example_position, example_segmentation, deterministic, model_mode)

    state_io, shift, circ_storage, circ_storage_mover = self.init_states(inputs)
    total_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats + self.num_stages  - 1 
    # TODO(huge): Shard the weights. This may be tricky b/c there is no "stage" axis in the weights to shard over until after the below
    weights = [decoder.variables for decoder in self.decoder_layers]
    # Go from a list of size n_layers of weight pytrees to a single pytree where each leaf has a leading dimension of n_layers 
    weights = stack_pytrees(*weights)
    breakpoint()
    # TODO: may want to have some simplified flow when is initializing instead c
    # (don't need to run through total_iters)
    for loop_iteration in range(total_iterations):
       print(f"starting iteration {loop_iteration}")
       state_io, shift, circ_storage, circ_storage_mover = self.run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, positions, segment_ids, deterministic, model_mode)

    # The final output is located in the input/output array, however the output microbatches may be permuted relative to the input
    final_output = self.permute_output_ms_dim(state_io)

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(final_output, (self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
                               
    return final_output