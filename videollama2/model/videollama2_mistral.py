# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, MistralConfig, MistralForCausalLM, MistralModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .videollama2_arch import Videollama2MetaForCausalLM, Videollama2MetaModel


class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"

    def __init__(
        self,
        use_fastv: bool = False,
        fastv_k: int = 3,
        fastv_r: float = 0.5,
        fastv_visual_token_start_index: Optional[int] = None,
        fastv_visual_token_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "videollama2_mistral"
        self.use_fastv = use_fastv
        self.fastv_k = fastv_k
        self.fastv_r = fastv_r
        self.fastv_visual_token_start_index = fastv_visual_token_start_index
        self.fastv_visual_token_length = fastv_visual_token_length


class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)
        self.fastv_visual_token_start_index = getattr(config, "fastv_visual_token_start_index", None)
        self.fastv_visual_token_length = getattr(config, "fastv_visual_token_length", None)

    def set_fastv_visual_token_span(self, start_index: Optional[int], token_length: Optional[int]):
        self.fastv_visual_token_start_index = start_index
        self.fastv_visual_token_length = token_length
        self.config.fastv_visual_token_start_index = start_index
        self.config.fastv_visual_token_length = token_length

    def clear_fastv_visual_token_span(self):
        self.set_fastv_visual_token_span(None, None)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        output_attentions: bool,
    ):
        if self._attn_implementation == "flash_attention_2":
            return attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        if self._attn_implementation == "sdpa" and not output_attentions:
            return _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        return _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    def _should_use_fastv(
        self,
        batch_size: int,
        hidden_states: torch.Tensor,
        past_key_values_length: int,
    ) -> bool:
        if not getattr(self.config, "use_fastv", False):
            return False
        if batch_size != 1:
            return False
        if hidden_states.shape[1] <= 1:
            return False
        if past_key_values_length != 0:
            return False
        start = self.fastv_visual_token_start_index
        length = self.fastv_visual_token_length
        if start is None or length is None:
            return False
        if int(length) <= 0:
            return False
        fastv_k = int(getattr(self.config, "fastv_k", 0) or 0)
        fastv_r = float(getattr(self.config, "fastv_r", 0.0) or 0.0)
        return fastv_k > 0 and 0.0 < fastv_r < 1.0

    def _fastv_keep_indices(self, last_layer_attention: torch.Tensor, seq_length: int) -> Optional[torch.LongTensor]:
        start = int(self.fastv_visual_token_start_index)
        length = int(self.fastv_visual_token_length)
        end = min(start + length, seq_length)
        if start < 0 or start >= end:
            return None

        fastv_r = float(getattr(self.config, "fastv_r", 0.0) or 0.0)
        keep_count = int(round((end - start) * (1.0 - fastv_r)))
        keep_count = max(1, min(end - start, keep_count))
        if keep_count >= (end - start):
            return None

        attn_avg = torch.mean(last_layer_attention, dim=1)[0]
        attn_last = attn_avg[-1]
        video_attn = attn_last[start:end]
        top_indices = video_attn.topk(keep_count).indices + start
        device = top_indices.device
        prefix = torch.arange(start, device=device)
        suffix = torch.arange(end, seq_length, device=device)
        return torch.cat((prefix, top_indices, suffix)).sort().values

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        use_legacy_cache = False
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        requested_output_attentions = bool(output_attentions)
        fastv_enabled = self._should_use_fastv(batch_size, inputs_embeds, past_key_values_length)
        prepared_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            batch_size,
            seq_length,
            inputs_embeds,
            past_key_values_length,
            requested_output_attentions or fastv_enabled,
        )

        hidden_states = inputs_embeds
        base_attention_mask = attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if requested_output_attentions else None
        next_decoder_cache = None
        last_fastv_attention = None

        fastv_k = int(getattr(self.config, "fastv_k", 0) or 0)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if fastv_enabled and layer_idx == fastv_k and last_fastv_attention is not None:
                keep_indices = self._fastv_keep_indices(last_fastv_attention, hidden_states.shape[1])
                if keep_indices is not None and keep_indices.numel() < hidden_states.shape[1]:
                    hidden_states = hidden_states[:, keep_indices, :]
                    position_ids = keep_indices.unsqueeze(0).to(position_ids.device)
                    if base_attention_mask is not None and base_attention_mask.dim() == 2:
                        base_attention_mask = base_attention_mask[:, keep_indices]
                    prepared_attention_mask = self._prepare_decoder_attention_mask(
                        base_attention_mask,
                        batch_size,
                        hidden_states.shape[1],
                        hidden_states,
                        0,
                        requested_output_attentions or fastv_enabled,
                    )

            layer_output_attentions = requested_output_attentions or (
                fastv_enabled and layer_idx == (fastv_k - 1)
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    prepared_attention_mask,
                    position_ids,
                    past_key_values,
                    layer_output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=prepared_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=layer_output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]

            if fastv_enabled and layer_idx == (fastv_k - 1):
                last_fastv_attention = layer_outputs[1]

            if requested_output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.model = Videollama2MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def set_fastv_config(self, use_fastv: bool, fastv_k: int, fastv_r: float):
        self.config.use_fastv = bool(use_fastv)
        self.config.fastv_k = int(fastv_k)
        self.config.fastv_r = float(fastv_r)
        self.get_model().config.use_fastv = self.config.use_fastv
        self.get_model().config.fastv_k = self.config.fastv_k
        self.get_model().config.fastv_r = self.config.fastv_r

    def clear_fastv_state(self):
        self.get_model().clear_fastv_visual_token_span()

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
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs.labels = labels

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs


AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)
