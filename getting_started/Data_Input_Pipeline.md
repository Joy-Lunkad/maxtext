<!--
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
## Data Input Pipeline

Currently MaxText supports three types of data input for training: HuggingFace datasets, Tensorflow Datasets (TFRecord files) through the tf.data based pipeline, and ArrayRecord files through the Grain pipeline for determinism. 

### HuggingFace datasets
The HuggingFace pipeline supports streaming directly from HuggingFace Hub, or from GCS bucket in HuggingFace supported formats (parquet, json, etc.). This is through the HuggingFace [`datasets.load_dataset` API](https://huggingface.co/docs/datasets/en/loading) with `streaming=True`, which take in `hf_*` parameters. 
#### Example config for streaming from HuggingFace Hub (no download needed):
```
dataset_type: hf
hf_path: 'allenai/c4'  # for using https://huggingface.co/datasets/allenai/c4
hf_data_dir: 'en'
hf_data_files: ''
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
hf_access_token: ''  # provide token if using gated dataset or tokenizer
```

#### Example config for streaming from downloaded data in a GCS bucket:
```
dataset_type: hf
hf_path: 'parquet'  # or json, arrow, etc.
hf_data_dir: ''
hf_data_files: 'gs://<bucket>/<folder>/*-train-*.parquet'  # match the train files
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
```
#### Limitations & Recommendations
1. Streaming data directly from HuggingFace Hub may be impacted by the traffic of the server. During peak hours you may encounter "504 Server Error: Gateway Time-out". It's recommended to download the HuggingFace dataset to a GCS buckt or disk for the most stable experience.
2. Streaming data directly from HuggingFace Hub works in multihost settings with a samll number of hosts. We have encountered "read time out" error with host number > 16.

### Deterministic Data Input Pipeline - Grain

MaxText users can optionally use [Grain](https://github.com/google/grain?tab=readme-ov-file), a deterministic data input pipeline. With Grain, the indexes of data trained are saved in a tiny json file in checkpoints, which allows you to keep the data order, restart from the exact same data, and reproduce the same losses. The whole training process becomes reproducible, disruption-proof, and debuggable. To use this pipeline:
1. Dataset needs to be in [ArrayRecord](https://github.com/google/array_record) format, which supports random access. For converting dataset into ArrayRecord, see [instructions](https://github.com/google/array_record/tree/main/beam).
2. ArrayRecord dataset, when hosted on GCS bucket, can only be read through [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse). The installation of Cloud Storage FUSE is included in [setup.sh](https://github.com/google/maxtext/blob/main/setup.sh). User then needs to mount the GCS bucket to a local path for each worker, using the script [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh). The script configs some parameters for the mount.
```
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$BUCKET_NAME MOUNT_PATH=$MOUNT_PATH
```
3. Set `dataset_type=c4-array_record` and set `dataset_path`, `dataset_name` accordingly. `dataset_path` should be the same as `$MOUNT_PATH` in the above step. `dataset_name` is the path to the folder that contains the ArrayRecord dataset, so that `os.path.join(config.dataset_path, config.dataset_name)` is the full path to the ArrayRecord files.
4. Tune `grain_worker_count` for performance. This parameter controls the number of child process used by Grain (more details in [behind_the_scene](https://github.com/google/grain/blob/main/docs/behind_the_scenes.md), [code](https://github.com/google/grain/blob/main/grain/_src/python/grain_pool.py)). If you use a large number of workers, please check your config for gcsfuse in [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh) to avoid gcsfuse throttling.
5. Example command:
```
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=/tmp/gcsfuse && python3 MaxText/train.py MaxText/configs/base.yml run_name=<RUN_NAME> base_output_directory=gs://<MY_BUCKET>  dataset_path=/tmp/gcsfuse/ dataset_name='array-record/c4/en/3.0.1' dataset_type=c4-array_record grain_worker_count=2
```

### Tensorflow Datasets

1. Download the Allenai c4 dataset in TFRecord format to a GCS bucket (will cost about $100, [details](https://github.com/allenai/allennlp/discussions/5056))
```
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```
2. Use the following config:
```
dataset_type: c4
dataset_name: 'c4/en:3.0.1'
tokenizer_path: "assets/tokenizer.llama2"
```
