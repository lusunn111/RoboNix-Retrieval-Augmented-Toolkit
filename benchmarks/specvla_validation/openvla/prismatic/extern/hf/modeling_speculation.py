import copy
import json
import time

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from openvla.specdecoding.model.cnets import MMModel,PMMModel
from openvla.specdecoding.model.cnets import EConfig
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import safetensors

from .utils import *
from .kv_cache import initialize_past_key_values

import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache

logger = logging.get_logger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

class LlamaSpecForCausalLM(LlamaForCausalLM):
    def __init__(self,config:LlamaConfig,attn_implementation):
        super().__init__(config=config)
        self.tree_mask = None
        return
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('customized tree mask')
        #print(attention_mask.shape)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        #print('past kv type',type(outputs.past_key_values))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        #labels: torch.LongTensor = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        #print('customized forward!!!!!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()
                #print('use old cache')
        #print('past seen tokens',past_seen_tokens)
        #print(past_seen_tokens)
        #if hasattr(self, "tree_mask") and self.tree_mask is not None:
        #    cache_position = position_ids
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        #print(cache_position)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        #else:
        #    cache_position = position_ids
        #print('cache position',cache_position)
        #print('past seen tokens',past_seen_tokens)
        #TODO:Update this function to fullfill the requirements.
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        #print('causal mack',causal_mask[0][0][-20:][:,-20:])
        #print(self.tree_mask)
        #print(position_ids)
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        #print(hidden_states.shape)
        #print(causal_mask.shape)
        #print('verify position ids',position_ids)
        #print()
        #print('model forward use cache')
        #print(use_cache)

        for decoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            #loss = None,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        #disable this feature
        #to specify the attention mask.
        '''if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                #print('return None')
                return None'''
        
        dtype, device = input_tensor.dtype, input_tensor.device
        #print(attention_mask.shape)
        #print(input_tensor.shape)
        #print('cache_position',cache_position.shape)
        #print('past see tokens',past_seen_tokens)
        min_dtype = torch.finfo(dtype).min
        #max_dtype = torch.finfo(dtype).max
        sequence_length = input_tensor.shape[1]
        #print('sequence length',sequence_length)
        #print('attention mask',attention_mask)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
            #print('static cache')
        else:  # dynamic cache
            if hasattr(self, "tree_mask") and self.tree_mask is not None:
                target_length=past_seen_tokens + sequence_length
            elif isinstance(attention_mask, torch.Tensor):
                target_length = attention_mask.shape[-1]
            else:
                target_length = past_seen_tokens + sequence_length + 1
            '''target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )'''
        #print('target length',target_length)

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):
            causal_mask = torch.triu(causal_mask, diagonal=1+past_seen_tokens)
        elif sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        #print('causal mask',causal_mask[:, -2])
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        #print('causal mask shape',causal_mask.shape)
        #print('causal mask num',causal_mask[0][0][-1])
        #print('attention mask shape',attention_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print('causal mask',causal_mask[:, :, -2])
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            #print('update based on attention mask')
            if attention_mask.dim() == 2:
                #print('dim = 2')
                mask_length = attention_mask.shape[-1]
                #print('mask length',mask_length)
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                #print(padding_mask)
                padding_mask = padding_mask == 0
                #print('padding mask',padding_mask)
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                #print('dim = 4')
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice
        #print('no tree mask')
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree_mask',tree_mask)
            tree_len = tree_mask.size(-1)
            causal_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = min_dtype
        #else:
        #    print('no tree mask')
        #print('final tree mask')
        #print(causal_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print(causal_mask.shape)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class SpecVLAforActionPrediction(nn.Module):
    '''def __init__(self,openvla=None,head=None):
        self.base_model = openvla
        self.ea_layer = head'''
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            parallel_draft=False,
            total_token=None,
            depth=None,
            top_k=None,
            threshold=None,
            accept_threshold=None
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.language_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.language_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        #if not parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
        config = EConfig.from_pretrained(ea_model_path)
        #else:
        #    config = EConfig.from_pretrained(ea_model_path)

        self.accept_threshold=accept_threshold
        #print('init accept threshold',accept_threshold)
        self.norm_stats = base_model.norm_stats

        # Compute action bins
        self.bins = base_model.bins
        self.bin_centers = base_model.bin_centers

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = base_model.vocab_size
        self.get_action_stats = base_model.get_action_stats
        if parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
            with open(ea_model_path+'/model.safetensors', "rb") as f:
                safetensors_model = f.read()
                #pytorch_model = safetensors.torch.deserialize(safetensors_model)
                #print(state_dict.keys())
                self.ea_layer = PMMModel(config, path=base_model_name_or_path,load_emb=True)
                ea_layer_state_dict = safetensors.torch.load(safetensors_model)
        else:
            self.ea_layer = MMModel(config, path=base_model_name_or_path,load_emb=True)
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path)
        #self.ea_layer.init_tree()
        self.tree_mask = None
        low_memory = False

        device = base_model.language_model.model.layers[-1].self_attn.q_proj.weight.device
        #load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.embed_tokens = self.base_model.language_model.model.embed_tokens
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.ea_layer.tree_mask = None
        self.ea_layer.tree_mode = None
        #print(self.ea_layer.fc.weight)
        #exit()
        #self.base_model.language_model = LlamaSpecForCausalLM(language_model)
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        return self.base_model.get_action_dim(unnorm_key)
    def forward(
        self,
        output_orig=False,
        input_embeds = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None
    ):
         #先这样，后面看结合需求怎么改，我的判断是得根据需要的数据模态把需要的内容放进去
         with torch.inference_mode():
            #reorganize the embeddings
            #print('forward not tested.')
            #print(output_hidden_states)
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values = pixel_values,
                labels = labels,
                inputs_embeds = input_embeds,
                past_key_values=past_key_values,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states=True,
                output_projector_features=output_projector_features,
                return_dict=return_dict,
                position_ids=position_ids
            )
            #print(outputs.keys())
            if output_orig:
                orig = outputs.logits
            #print(len(outputs.hidden_states))
            hidden_states = outputs.hidden_states[-1]
            input_embeddings = outputs.hidden_states[0]
            #print(len(hidden_states))
            #print(torch.cat(hidden_states).shape)
            if output_orig:
                return outputs, orig, hidden_states,input_embeddings
            else:
                return outputs, hidden_states
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        unnorm_key: Optional[str] = None,
        return_hidden_states: bool = False,
        legacy_output_hidden: Optional[bool] = None,
        generate_mode = 'Speculative',
        return_sd_stats: bool = False,  # 是否返回 SD 统计信息
        #accept_threshold=None,
        **kwargs: str
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[torch.FloatTensor]]]:
        """Wrapper around .generate() that decodes predicted actions and can return hidden states.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalizing actions
            return_hidden_states: Whether to return the last hidden state
            legacy_output_hidden: Legacy parameter, equivalent to return_hidden_states
            **kwargs: Additional arguments for generate

        Returns:
            If return_hidden_states=False:
                unnormalized actions as numpy array
            Otherwise:
                Tuple of (unnormalized_actions, hidden_states)
        """
        # 处理参数，支持旧的参数命名方式
        if legacy_output_hidden is not None:
            return_hidden_states = legacy_output_hidden

        # 设置generate方法的参数
        if return_hidden_states:
            kwargs['output_hidden_states'] = True
        
        # 如果特殊的空标记不在提示末尾，则添加它
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            #print('add special token')
            #print(kwargs['attention_mask'])
            kwargs['attention_mask']=torch.cat(
                (kwargs['attention_mask'], torch.unsqueeze(torch.Tensor([1]), dim=0).to(input_ids.device)), dim=1
            ).to(int)
            #print(kwargs['attention_mask'])
        #print(kwargs)
        #exit()
        # 运行模型生成
        #print('base model generate')
        '''outputs = self.ea_forward(
            input_ids=input_ids,
            max_new_tokens=self.get_action_dim(unnorm_key),
            #return_dict=True,
            #return_dict_in_generate=True,
            **kwargs
        )'''
        sd_stats = None  # SD 统计信息
        if str(generate_mode).lower() == 'speculative':
            outputs, sd_stats = self.eagenerate(
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                #return_dict=True,
                #return_dict_in_generate=True,
                accept_threshold=self.accept_threshold,
                **kwargs
            )
            #print(outputs)
        else:
            #print('ea_forward')
            outputs = self.ea_forward(
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                #return_dict=True,
                #return_dict_in_generate=True,
                **kwargs
                )
        #print(outputs)
        #print(outputs.hidden_states[0][0].shape)
        #input()
        #print(outputs.shape)
        #print(-self.get_action_dim(unnorm_key))
       # exit()
        #print("LOCAL_TRANSFORMER.generate方法!!!!!!!")
        #print(outputs)
        # 获取生成的token IDs
        #print(generated_ids)
        if hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
            #eagle_generated_ids = eagle_outputs.sequences
        elif len(outputs)==2:
            generated_ids = outputs[0]
        else:
            generated_ids = outputs
            #eagle_generated_ids = eagle_outputs.sequences
        #print('generate ids',generated_ids)
        #print('output sequence',outputs)
        #print(-self.get_action_dim(unnorm_key))
        #exit()
        #print(generated_ids)
        # 从生成的tokens转换为动作值
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):].cpu().numpy()
        #print('predicted_ids',predicted_action_token_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        #print('vocab_size',self.vocab_size)
        #print('final_action_token',discretized_actions)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        
        normalized_actions = self.bin_centers[discretized_actions]
       # print(normalized_actions) 
        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        # 如果需要返回隐藏状态
        if return_hidden_states:
            if (len(outputs)==2):
                tmp_hidden = outputs[1]
            else:
                tmp_hidden=outputs.hidden_states
            # 使用前向传播获取隐藏状态
            first_layer_hidden = []
            last_layer_hidden = []
            #print(len(outputs.hidden_states))
            for i in range(len(tmp_hidden)):
                last_layer_hidden.append(tmp_hidden[i][-1].cpu()[0])
                first_layer_hidden.append(tmp_hidden[i][0].cpu()[0])
            # 返回二元组: (动作, 隐藏状态)
            #print(last_layer_hidden[0].shape)
            return actions, predicted_action_token_ids,(first_layer_hidden,last_layer_hidden)
        
        # 如果需要返回 SD 统计信息
        if return_sd_stats and sd_stats is not None:
            return actions, sd_stats
        
        # 否则只返回动作
        return actions
    @torch.no_grad()
    def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder
    ):
        #print(model_kwargs.keys())
        #print(outputs.keys())
        #exit()
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs
        )
        #print(model_kwargs["past_key_values"][0][0].shape)
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                #print('update attention mask')
                #print(attention_mask)
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            #else:
            #    print('no attention mask to update')
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs
    @torch.no_grad()
    def ea_forward(self,input_ids,max_new_tokens, logits_processor=None,output_hidden_states=False,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['attention_mask'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        if output_hidden_states:
            hidden_states = []
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states
            )
        if output_hidden_states:
            hidden_states.append(outputs.hidden_states)
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        #print(type(outputs['past_key_values']))
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        model_inputs['attention_mask']=outputs.attention_mask
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            #print(model_inputs)
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                #use_cache = True
            )
            if output_hidden_states:
                hidden_states.append(outputs.hidden_states)
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        #print('ea forward',output_hidden_states)
        if output_hidden_states:
            #print(outputs.hidden_states)
            return input_ids[:,input_len+1:],hidden_states[:-1]
        return input_ids[:,input_len+1:]
    def ea_forward_embed(self,input_ids,max_new_tokens, logits_processor=None,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                #use_cache = True
            )
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        return input_ids[:,input_len:]
    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        max_new_tokens,
        #return_dict=True,
        #return_dict_in_generate=True,
        log = False,
        accept_threshold=None,
        **kwargs
    ):
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        reset_tree_mode(self.ea_layer)
        time_0 = time.time()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, prompt_hidden_states, sample_token, past_key_value_data,prompt_embeds,attention_mask = initialize_tree(model_inputs, self, logits_processor)
        input_len = input_ids.shape[1]-1
        max_length = max_length - self.ea_layer.total_tokens - 10
        new_token = 0
        for idx in range(max_length):
            # with Timer("all"):
            cycle_begin_time = time.time()
            self.base_model.language_model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new,hidden_embedding_new,past_kv_data_new,outputs= tree_decoding(
                self,
                prompt_embeds,
                draft_tokens,
                attention_mask,
                past_key_value_data,
                tree_position_ids,
                #input_ids,
                retrieve_indices,
                #draft_logit=draft_logit
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor,accept_threshold=accept_threshold
            )
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token,prompt_embeds,past_key_value_data,attention_mask = update_inference_inputs(
                prompt_embeds,
                #prompt_hidden_states,
                input_ids,
                input_len,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_kv_data_new,
                #current_length_data,
                self,
                hidden_state_new,
                #hidden_embedding_new,
                sample_p,
                attention_mask
            )
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        #print('end loop')
        #print('check stop tokens')
        stop_token_ids_index = [
                    i
                    for i, id in enumerate(input_ids[0])
                    if (id == self.tokenizer.eos_token_id or id == self.tokenizer.pad_token_id)
                ]
        if len(stop_token_ids_index) > 0:
                    input_ids = input_ids[:,:stop_token_ids_index[0]]

        if not log:
            # 返回 tokens 和 SD 统计信息（new_token=总生成token数, idx+1=迭代次数）
            # 平均接受长度 = new_token / (idx + 1)
            sd_stats = {'new_token': new_token, 'num_iterations': idx + 1}
            return input_ids[:,input_len+1:], sd_stats
        else:
            return input_ids, new_token, idx
    def eval_topk(self,input_ids, logits_processor=None,**kwargs):
        #token = torch.tensor(token).to(input_ids.device)
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        #self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = 6
        #model = self
        #print('base model')
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #这里直接用ea_forward那最后一个位置的hidden state
        #print(kwargs)
        kwargs['return_hidden_states']=True
        #print(kwargs)
        #exit()
        action,tokens,hidden = self.predict_action(
               input_ids, logits_processor=None,**kwargs
            )
        token = torch.tensor(tokens).to(input_ids.device)
        input_embeds = hidden[0]
        hidden_states = hidden[1]
        hidden_states = torch.cat([item for item in hidden[1]],dim=0).to(input_ids.device)
        input_embeds = torch.cat([item for item in hidden[0]],dim=0).to(input_ids.device)
        #print(input_embeds.device)
        input_token_embeds = self.ea_layer.embed_tokens(torch.tensor([2]).to(input_ids.device))
        ea_layer_input_embeds = torch.cat((input_embeds,input_token_embeds),dim=0)
        self.ea_layer._eval_top_k(hidden_states,token, ea_layer_input_embeds,self.base_model.language_model.lm_head, 0,logits_processor)
        return action,None,None

    @torch.no_grad()
    def block_sd_verify(
        self,
        input_ids,
        top_k_tokens: torch.Tensor,  # [K, 7] draft tokens from retrieval
        blocks: list,  # [{'name': 'position', 'indices': [0,1,2]}, ...]
        prob_threshold: float = 0.001,  # 保留但不再使用
        use_avg_prob: bool = True,  # 保留但不再使用
        accept_threshold: int = None,  # 保留但不再使用
        block_thresholds: dict = None,  # 保留但不再使用
        block_sum_threshold: int = 45,  # α: 总差值阈值 (放松)
        block_max_threshold: int = 25,  # μ: 单个差值阈值 (放松)
        **kwargs
    ):
        """
        Block-wise Speculative Decoding with retrieved candidates
        
        一次 forward 验证所有 7 个 tokens（K 条链并行）！
        然后逐 block 用 token 差值判断是否通过，失败的 block 开始 AR 生成。
        
        接受条件（两个都要满足）：
        1. sum(|candidate_token - argmax_token|) < block_sum_threshold (α)
        2. max(|candidate_token - argmax_token|) < block_max_threshold (μ)
        
        Args:
            input_ids: Input token ids (prompt)
            top_k_tokens: [K, 7] Draft tokens from retrieval
            blocks: List of block configs
            prob_threshold: Joint probability threshold
            use_avg_prob: Use geometric mean probability
            accept_threshold: Token difference tolerance
            block_thresholds: Per-block probability thresholds
        
        Returns:
            final_tokens: [7] Final action tokens
            stats: Verification statistics
        """
        import math
        
        K = top_k_tokens.shape[0]
        device = input_ids.device
        num_tokens = 7  # 总共 7 个 action tokens
        
        # 清除 tree_mask
        self.tree_mask = None
        self.base_model.language_model.tree_mask = None
        
        # ========================================
        # Step 1: Prefill 获取 context
        # ========================================
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        model_inputs.pop('use_cache', None)
        model_inputs.pop('return_dict', None)
        
        outputs = self.base_model(
            **model_inputs,
            return_dict=True,
            use_cache=True,
        )
        
        prefill_logits = outputs.logits[0, -1, :]  # [vocab]
        current_past_kv = outputs.past_key_values
        context_len = current_past_kv[0][0].shape[2]
        
        # ========================================
        # 快速路径: 如果 token 差值阈值 < 0，跳过 tree_verify，直接 AR
        # ========================================
        if block_sum_threshold < 0 or block_max_threshold < 0:
            stats = {'blocks': [], 'accepted': False, 'mode': 'pure_AR', 'skip_tree_verify': True, 'accepted_tokens': 0}
            final_tokens = []
            ar_past_kv = current_past_kv
            ar_logits = prefill_logits
            
            for i in range(num_tokens):
                next_token = ar_logits.argmax().item()
                final_tokens.append(next_token)
                if i < num_tokens - 1:
                    t_tensor = torch.tensor([[next_token]], device=device)
                    t_embed = self.base_model.language_model.model.embed_tokens(t_tensor)
                    out = self.base_model.language_model(
                        inputs_embeds=t_embed,
                        past_key_values=ar_past_kv,
                        use_cache=True,
                        return_dict=True
                    )
                    ar_past_kv = out.past_key_values
                    ar_logits = out.logits[0, -1, :]
            
            return torch.tensor(final_tokens, dtype=torch.long), stats
        
        # ========================================
        # Step 2: 一次 forward 验证所有 K 条链的 7 个 tokens
        # ========================================
        # 构建 tree attention mask: K 条链并行，链内 causal，链间不可见
        # tree_mask 形状应为 [1, 1, total_len, total_len]，参考 generate_tree_buffers
        total_len = K * num_tokens
        tree_mask = torch.zeros(total_len, total_len, device=device)
        
        for k in range(K):
            start = k * num_tokens
            for i in range(num_tokens):
                for j in range(i + 1):
                    tree_mask[start + i, start + j] = 1
        
        # 加两个维度变成 [1, 1, total_len, total_len]
        tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)
        
        # 设置 tree_mask
        self.tree_mask = tree_mask
        self.base_model.language_model.tree_mask = tree_mask
        
        # 构建 position_ids: 同一位置的不同候选共享 position
        position_ids = torch.zeros(1, total_len, dtype=torch.long, device=device)
        for k in range(K):
            for i in range(num_tokens):
                position_ids[0, k * num_tokens + i] = context_len + i
        
        # 将 K 条链展平 [1, K * 7]
        draft_flat = top_k_tokens.to(device).flatten().unsqueeze(0)
        draft_embeds = self.base_model.language_model.model.embed_tokens(draft_flat)
        
        # 调用 self() 而不是 self.base_model.language_model()
        # 参考 SpecVLA tree_decoding 的调用方式，必须用 use_cache=True
        outputs_verify, tree_logits, _, _ = self(
            input_embeds=draft_embeds,
            output_orig=True,
            attention_mask=None,
            past_key_values=current_past_kv,
            return_dict=True,
            position_ids=position_ids,
            use_cache=True
        )
        
        self.tree_mask = None
        self.base_model.language_model.tree_mask = None
        
        # all_logits: [K * 7, vocab]
        all_logits = tree_logits.squeeze(0)
        
        # ========================================
        # Step 3: 计算每个位置的 argmax token 和候选 token 差值
        # ========================================
        # 获取每个位置的 argmax token
        argmax_tokens = []
        for i in range(num_tokens):
            if i == 0:
                logits_i = prefill_logits
            else:
                logits_i = all_logits[i - 1]  # 第一条链的对应位置
            argmax_tokens.append(logits_i.argmax().item())
        
        # 计算每个候选每个 token 的差值
        all_token_diffs = {}  # {k: [diff0, diff1, ..., diff6]}
        for k in range(K):
            token_diffs = []
            for i in range(num_tokens):
                candidate_token = top_k_tokens[k, i].item()
                diff = abs(candidate_token - argmax_tokens[i])
                token_diffs.append(diff)
            all_token_diffs[k] = token_diffs
        
        # ========================================
        # Step 4: 先判断所有 Block（不更新 KV cache）
        # ========================================
        stats = {'blocks': [], 'argmax_tokens': argmax_tokens}
        
        alive_candidates = list(range(K))
        first_failed_block_idx = None
        block_results = []  # 存储每个 block 的判断结果
        
        for block_idx, block in enumerate(blocks):
            block_name = block['name']
            block_indices = block['indices']
            
            # 计算存活候选在该 block 的差值统计
            candidate_diffs = {}
            for k in alive_candidates:
                block_diffs = [all_token_diffs[k][i] for i in block_indices]
                candidate_diffs[k] = {
                    'diffs': block_diffs,
                    'sum': sum(block_diffs),
                    'max': max(block_diffs),
                }
            
            # 找出通过条件的候选
            passed_candidates = [
                k for k, d in candidate_diffs.items()
                if d['sum'] < block_sum_threshold and d['max'] < block_max_threshold
            ]
            
            # 找最佳候选
            best_k = min(candidate_diffs.keys(), key=lambda k: candidate_diffs[k]['sum'])
            
            block_stat = {
                'name': block_name,
                'sum_threshold': block_sum_threshold,
                'max_threshold': block_max_threshold,
                'best_candidate': best_k,
                'best_sum_diff': candidate_diffs[best_k]['sum'],
                'best_max_diff': candidate_diffs[best_k]['max'],
                'alive_before': list(alive_candidates),
                'passed_candidates': passed_candidates,
                'all_diffs': {k: candidate_diffs[k] for k in alive_candidates},
            }
            
            if len(passed_candidates) > 0:
                alive_candidates = passed_candidates
                block_stat['mode'] = 'verified'
                block_stat['alive_after'] = list(alive_candidates)
            else:
                block_stat['mode'] = 'AR'
                block_stat['alive_after'] = []
                if first_failed_block_idx is None:
                    first_failed_block_idx = block_idx
                alive_candidates = list(range(K))  # 重置
            
            block_results.append((block_stat, best_k))
            stats['blocks'].append(block_stat)
        
        # ========================================
        # Step 5: 根据判断结果生成 final_tokens
        # ========================================
        if first_failed_block_idx is None:
            # 全部通过！直接使用最佳候选的 tokens，无需 KV 更新
            best_k = block_results[0][1]  # 第一个 block 的最佳候选
            final_tokens = top_k_tokens[best_k].tolist()
            stats['accepted'] = True
            stats['mode'] = 'fully_verified'
            stats['num_ar_blocks'] = 0
            stats['accepted_tokens'] = num_tokens  # 全部 7 个 token 被接受
        else:
            # 有 block 失败，需要 AR
            final_tokens = []
            ar_past_kv = current_past_kv
            ar_logits = prefill_logits
            
            for block_idx, block in enumerate(blocks):
                block_indices = block['indices']
                block_stat = block_results[block_idx][0]
                best_k = block_results[block_idx][1]
                
                if block_idx < first_failed_block_idx:
                    # 这个 block 之前已验证通过，使用检索结果
                    # 但需要更新 KV cache 供后续 AR 使用
                    for token_idx in block_indices:
                        t = top_k_tokens[best_k, token_idx].item()
                        final_tokens.append(t)
                        
                        t_tensor = torch.tensor([[t]], device=device)
                        t_embed = self.base_model.language_model.model.embed_tokens(t_tensor)
                        out = self.base_model.language_model(
                            inputs_embeds=t_embed,
                            past_key_values=ar_past_kv,
                            use_cache=True,
                            return_dict=True
                        )
                        ar_past_kv = out.past_key_values
                        ar_logits = out.logits[0, -1, :]
                else:
                    # 从第一个失败的 block 开始，全部用 AR
                    for token_idx in block_indices:
                        next_token = ar_logits.argmax().item()
                        final_tokens.append(next_token)
                        
                        if token_idx < num_tokens - 1:
                            t_tensor = torch.tensor([[next_token]], device=device)
                            t_embed = self.base_model.language_model.model.embed_tokens(t_tensor)
                            out = self.base_model.language_model(
                                inputs_embeds=t_embed,
                                past_key_values=ar_past_kv,
                                use_cache=True,
                                return_dict=True
                            )
                            ar_past_kv = out.past_key_values
                            ar_logits = out.logits[0, -1, :]
            
            stats['accepted'] = False
            stats['mode'] = 'partial_AR'
            stats['first_failed_block'] = first_failed_block_idx
            stats['num_ar_blocks'] = len(blocks) - first_failed_block_idx
            # 计算被接受的 token 数 = first_failed_block 之前所有 block 的 token 数
            accepted_tokens = sum(len(blocks[i]['indices']) for i in range(first_failed_block_idx))
            stats['accepted_tokens'] = accepted_tokens
        
        self.base_model.language_model.tree_mask = None
        
        return torch.tensor(final_tokens, dtype=torch.long), stats
