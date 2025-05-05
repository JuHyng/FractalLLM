import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import evaluate  # pip install evaluate
# Evaluate metrics

from typing import Optional, List, Tuple, Dict, Any
from transformers.cache_utils import Cache

from transformers.models.llama.modeling_llama import LlamaAttention


bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


def gumbel_softmax(logits, tau=1.0, hard=False):
    device = logits.device
    gumbels = -torch.empty_like(logits, device=device).exponential_().log()
    y = logits + gumbels
    y = F.softmax(y / tau, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y, device=device).scatter_(
            -1, y.argmax(dim=-1, keepdim=True), 1.0
        )
        y = (y_hard - y).detach() + y

    return y

class FlopsCounter:
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Hook the linear layers
                handle = module.register_forward_hook(self._linear_flops_hook)
                self.handles.append(handle)
            elif isinstance(module, LlamaAttention):
                # Hook the custom LLaMA attention
                handle = module.register_forward_hook(self._llama_attn_flops_hook)
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.total_flops = 0

    def get_total_flops(self):
        return self.total_flops

    def _linear_flops_hook(self, module, input, output):
        
        x = input[0]
        if x.dim() == 3:
            batch_size, seq_len, in_features = x.shape
        elif x.dim() == 2:
            # no seq dimension (possibly intermediate usage),
            # treat seq_len=1 for flop count
            batch_size, in_features = x.shape
            seq_len = 1
        else:
            # fallback
            batch_size = x.shape[0]
            seq_len = 1
            in_features = module.in_features
        
        out_features = module.out_features
        # Multiply-add pairs => *2. Adjust if you prefer otherwise
        flops = 2.0 * batch_size * seq_len * in_features * out_features
        self.total_flops += flops

    def _llama_attn_flops_hook(self, module, input, output):

        attn_output = output[0]  # shape [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = attn_output.shape
        if hasattr(module, "num_heads"):
            num_heads = module.num_heads
        elif hasattr(module, "config") and hasattr(module.config, "num_attention_heads"):
            num_heads = module.config.num_attention_heads
        else:
            raise ValueError("Cannot find `num_heads` or `num_attention_heads` in LlamaAttention.")

        head_dim = hidden_dim // num_heads
        
        # QK^T cost:
        qk_flops = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        # attn_weights @ V cost:
        av_flops = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        self.total_flops += (qk_flops + av_flops)


def measure_generation_time(model, input_ids, max_length=50):
    
    start = time.time()
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=max_length)
    end = time.time()

    total_time = end - start
    new_tokens = out.shape[-1] - input_ids.shape[-1]  
    
    if new_tokens == 0:
        time_per_new_token = 0.0
    else:
        time_per_new_token = total_time / new_tokens
    
    return out, total_time, time_per_new_token

def compute_metrics(prediction: str, reference: str):
    bleu_results = bleu_metric.compute(predictions=[prediction], references=[[reference]])
    bleu_score = bleu_results["bleu"]
    rouge_results = rouge_metric.compute(predictions=[prediction], references=[reference])
    rouge1_score = rouge_results["rouge1"]
    rouge2_score = rouge_results["rouge2"]
    rougeL_score = rouge_results["rougeL"]
    return bleu_score, rouge1_score, rouge2_score, rougeL_score


def compute_causal_mask(
    attention_mask: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool,
    config,
    training: bool,
) -> Optional[torch.Tensor]:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.cache_utils import StaticCache

    if config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    if config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]
    target_length = (
        attention_mask.shape[-1]
        if attention_mask is not None
        else past_seen_tokens + sequence_length + 1
    )

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


class FractalCache(Cache):

    def __init__(self, draft_layer_indexes, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: Dict[List[torch.Tensor]] = {}
        self.value_cache: Dict[List[torch.Tensor]] = {}
        self._fractal_index = 0
        self.draft_layer_indexes = draft_layer_indexes
        for draft_layer in draft_layer_indexes:
            self.key_cache[draft_layer] = []
            self.value_cache[draft_layer] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[self._fractal_index][layer_idx], self.value_cache[self._fractal_index][layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[self._fractal_index][layer_idx], self.value_cache[self._fractal_index][layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache[self._fractal_index])
    
    def set_fractal_index(self, fractal_index: int):
        self._fractal_index = fractal_index

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache[self._fractal_index]) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache[self._fractal_index]), layer_idx):
                    self.key_cache[self._fractal_index].append([])
                    self.value_cache[self._fractal_index].append([])
                self.key_cache[self._fractal_index].append(key_states)
                self.value_cache[self._fractal_index].append(value_states)
            elif (
                len(self.key_cache[self._fractal_index][layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[self._fractal_index][layer_idx] = key_states
                self.value_cache[self._fractal_index][layer_idx] = value_states
            else:
                self.key_cache[self._fractal_index][layer_idx] = torch.cat([self.key_cache[self._fractal_index][layer_idx], key_states], dim=-2)
                self.value_cache[self._fractal_index][layer_idx] = torch.cat([self.value_cache[self._fractal_index][layer_idx], value_states], dim=-2)

        return self.key_cache[self._fractal_index][layer_idx], self.value_cache[self._fractal_index][layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        
        # DEBUG
        print("self._fractal_index", self._fractal_index)
        
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache[self._fractal_index]) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[self._fractal_index][layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[self._fractal_index][layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def set_verify_mode(self, is_verify):
        self.is_verify = is_verify
        
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for fractal_index in self.key_cache.keys():
            for idx in range(len(self.key_cache)):
                if self.key_cache[fractal_index][idx] != []:
                    self.key_cache[fractal_index][idx] = self.key_cache[fractal_index][idx][..., :max_length, :]
                    self.value_cache[fractal_index][idx] = self.value_cache[fractal_index][idx][..., :max_length, :]
                
                
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["FractalCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for fractal_index in self.key_cache.keys():
            for i in range(0, full_batch_size, split_size):
                current_split = FractalCache(draft_layer_indexes=self.draft_layer_indexes)
                current_split._seen_tokens = self._seen_tokens
                current_split.key_cache[fractal_index] = [tensor[i : i + split_size] for tensor in self.key_cache[fractal_index]]
                current_split.value_cache[fractal_index] = [tensor[i : i + split_size] for tensor in self.value_cache[fractal_index]]
                out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["FractalCache"], num_hidden_layers: int = None) -> "FractalCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for fractal_index in splits[0].key_cache.keys():
            for idx in range(len(splits[0])):
                key_cache = [current.key_cache[fractal_index][idx] for current in splits if current.key_cache[fractal_index][idx] != []]
                value_cache = [current.value_cache[fractal_index][idx] for current in splits if current.value_cache[fractal_index][idx] != []]
                if key_cache != []:
                    layer_keys = torch.cat(key_cache, dim=0)
                    layer_values = torch.cat(value_cache, dim=0)
                    cache.update(layer_keys, layer_values, idx)
        return cache