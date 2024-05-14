"""
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
"""

"""Operations used by HuggingFace input pipeline"""


import dataclasses
import numpy as np
from threading import current_thread
from datasets import IterableDataset
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import max_logging


def tokenization(example, tokenizer, max_length):
  """Tokenize HF dataset"""
  return tokenizer(example["text"], truncation=True, max_length=max_length)


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys from hf input"""

  def map(self, features):
    return {
        "inputs": np.asarray(features["input_ids"], dtype=np.int32),
        "targets": np.asarray(features["input_ids"], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(self, dataset: IterableDataset, dataloading_host_index: int, dataloading_host_count: int, num_threads: int):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = None

  def _check_shard_count(self):
    if self.n_shards % (self.dataloading_host_count * self.num_threads) != 0:
      usable_shards = (
          self.n_shards
          // (self.dataloading_host_count * self.num_threads)
          * (self.dataloading_host_count * self.num_threads)
      )
      max_logging.log(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used.")
      max_logging.log("Make (dataset shards) % (number of host loading data) == 0 to use all shards of data")

  def _update_shard(self, idx):
    max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
    self.dataset_shards[idx] += self.dataloading_host_count * self.num_threads
    max_logging.log(f"New shard is {self.dataset_shards[idx]}")
    if self.dataset_shards[idx] > self.n_shards:
      raise ValueError(f"Run out of shards, shard {self.dataset_shards[idx]} is not available")
    self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
    self.data_iters[idx] = iter(self.datasets[idx])

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if self.data_iters is None:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)
