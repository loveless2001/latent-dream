import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kv_dreaming.runtime import QwenRuntime


@pytest.fixture
def runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=1024, bos_token_id=1, eos_token_id=2,
                         attention_dropout=0.0, rope_theta=1000000.0)
    config._attn_implementation = "eager"
    return QwenRuntime(Qwen3ForCausalLM(config))

