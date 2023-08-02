#!/bin/bash
set -e



prng_key=$1
run_name=$2


warmup_steps=1000 # batch of 4 => 19k total steps, 1k ~5%

output_file=gs://mattdavidow-maxtext-br/${run_name}.txt

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=3001 per_device_batch_size=4 learning_rate=0.001 warmup_steps=${warmup_steps} enable_profiler=false enable_checkpointing=true \
    enable_dropout=false enable_data_shuffling=false run_name=${run_name}\
    base_output_directory=gs://maxtext-experiments-multipod\
    dataset_path=gs://max-datasets-rogue\
    int8_training=true metrics_file=metrics.txt\
    remat_policy=full init_prng_key=${prng_key}\
    fwd_int8=true bwd_int8=true"

echo "Starting run (${run_name}) with command: ${command}"
eval ${command}
echo "Finished command"
echo "Now writing to ${output_file}"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt ${output_file}
fi
echo "Done writing to ${output_file}"