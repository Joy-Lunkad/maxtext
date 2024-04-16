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

'''CLI Utility for Running Inference on a Single Stream'''

import jax

import maxengine
from jetstream.engine import token_utils

import os
import pyconfig
import sys


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  text = config.prompt
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  unpadded_tokens, unpadded_true_length = token_utils.tokenize_and_maybe_pad(text, vocab, 
                                                                         is_bos=True, pad_result=False, 
                                                                         prefill_lengths=None)
  # breakpoint()
  print(f"{unpadded_tokens=}")
  padded_prefill_result = engine.prefill(
      params=params, padded_tokens=unpadded_tokens, true_length=unpadded_true_length
  )
  # padded_prefill_result is a dictionary created using the vars from prefill():
  #         {"logits" : selected_logits, 
  #          "cache" : new_vars['cache'],
  #          "next_pos" : next_pos, 
  #          "generated_tokens" : generated_tokens}
  # 
  # padded_prefill_result: dict_keys(['cache', 'generated_tokens', 'logits', 'next_pos'])
  # padded_prefill_result['cache']['decoder'] = dict of layers 0-31
  # padded_prefill_result['logits'].shape = (1, 1, 32000) 
  # padded_prefill_result['next_pos'] = Array([[4]], dtype=int32)
  # padded_prefill_result['next_pos'].shape = (1, 1)
  # padded_prefill_result['generated_tokens'] = Array([[0]], dtype=int32)
  # padded_prefill_result['generated_tokens'].shape = (1, 1)
  assert unpadded_tokens.size <= config.max_prefill_predict_length, "can't take too many tokens"

  slot=0

  # breakpoint()
  decode_state = engine.init_decode_state()
  # Copies data from the prefill result into decode_state
  decode_state = engine.insert(
      padded_prefill_result, decode_state, slot=slot
  )

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer.detokenize(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert output==config.autoregressive_decode_assert, \
    f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"

def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  main(cfg)
