"""Frozen Qwen3 single-token forward with explicit cache and additive bias.

Native embedding, norms, projections, MLPs, and output head are reused. The
attention implementation is identical for eviction and merging.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from .cache import KVState

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
INSTRUCT_MODEL = "Qwen/Qwen3-0.6B"
INSTRUCT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAIN_MODEL = "Qwen/Qwen3-4B-Base"
MAIN_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
MODEL_PRESETS = {"qwen3-0.6b-base": (BASE_MODEL, BASE_REVISION),
                 "qwen3-4b-base": (MAIN_MODEL, MAIN_REVISION)}
DELIMITERS = {"<|im_start|>": 151644, "<|im_end|>": 151645,
              "<think>": 151667, "</think>": 151668}
STOP_ID = 151643


class QwenRuntime:
    def __init__(self, model, tokenizer=None):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        cfg = model.config
        if cfg.model_type != "qwen3" or cfg.rope_scaling or getattr(cfg, "use_sliding_window", False):
            raise ValueError("only unscaled full-attention Qwen3 is supported")
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.theta = float(cfg.rope_theta)
        self.inv_freq = model.model.rotary_emb.inv_freq.detach().float().clone()
        self.head_dim = cfg.head_dim
        self.kv_heads = cfg.num_key_value_heads
        if tokenizer is not None:
            for token, expected in DELIMITERS.items():
                if tokenizer.convert_tokens_to_ids(token) != expected:
                    raise ValueError(f"unexpected tokenizer mapping for {token}")
            if tokenizer.convert_tokens_to_ids("<|endoftext|>") != STOP_ID:
                raise ValueError("unexpected stop token")
            if cfg.bos_token_id != STOP_ID:
                raise ValueError("unexpected bootstrap token")

    @classmethod
    def load(cls, *, model_id=BASE_MODEL, revision=BASE_REVISION,
             dtype="float32", device="cpu", local_files_only=False):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("model revision must be a pinned 40-character commit")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager", local_files_only=local_files_only).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                  local_files_only=local_files_only)
        return cls(model, tokenizer)

    def empty(self, bootstrap_position=0):
        shape = (1, self.kv_heads, 0, self.head_dim)
        return KVState([torch.empty(shape, dtype=self.dtype, device=self.device)
                        for _ in self.model.model.layers],
                       [torch.empty(shape, dtype=self.dtype, device=self.device)
                        for _ in self.model.model.layers], bootstrap_position=bootstrap_position)

    @torch.inference_mode()
    def step(self, token_id: int, state: KVState):
        p = state.next_position
        if p >= self.model.config.max_position_embeddings:
            raise ValueError("position exceeds the checkpoint's configured range")
        ids = torch.tensor([[token_id]], device=self.device)
        h = self.model.model.embed_tokens(ids)
        position_ids = torch.tensor([[p]], device=self.device)
        cos, sin = self.model.model.rotary_emb(h, position_ids)
        # Each current token attends all past physical slots and itself.
        # Logical positions affect RoPE; source counts affect attention mass.
        bias = torch.tensor(state.counts + [1], device=self.device, dtype=torch.float32).log()
        for index, layer in enumerate(self.model.model.layers):
            residual = h
            x = layer.input_layernorm(h)
            a = layer.self_attn
            q = a.q_norm(a.q_proj(x).view(1, 1, -1, self.head_dim)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(1, 1, -1, self.head_dim)).transpose(1, 2)
            v = a.v_proj(x).view(1, 1, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            state.keys[index] = torch.cat((state.keys[index], k), dim=2)
            state.values[index] = torch.cat((state.values[index], v), dim=2)
            keys = repeat_kv(state.keys[index], a.num_key_value_groups)
            values = repeat_kv(state.values[index], a.num_key_value_groups)
            scores = torch.matmul(q, keys.transpose(2, 3)) * a.scaling
            weights = torch.softmax(scores.float() + bias[None, None, None, :], dim=-1).to(q.dtype)
            out = torch.matmul(weights, values).transpose(1, 2).reshape(1, 1, -1).contiguous()
            h = residual + a.o_proj(out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        state.positions.append(float(p))
        state.counts.append(1)
        state.next_position += 1
        logits = self.model.lm_head(self.model.model.norm(h))[0, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite model logits")
        return logits

    @torch.inference_mode()
    def prefill(self, *, token_ids=None, embeddings=None):
        if (token_ids is None) == (embeddings is None):
            raise ValueError("provide exactly one prefix representation")
        if token_ids is not None:
            inputs = {"input_ids": torch.tensor([token_ids], device=self.device)}
            n = len(token_ids)
        else:
            inputs = {"inputs_embeds": embeddings.to(self.device, self.dtype)}
            n = embeddings.shape[1]
        if n < 1 or n >= self.model.config.max_position_embeddings:
            raise ValueError("invalid prefill length")
        output = self.model(**inputs, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        keys = [layer.keys.detach() for layer in cache.layers]
        values = [layer.values.detach() for layer in cache.layers]
        return KVState(keys, values, [float(i) for i in range(n)], [1] * n, n, n)
