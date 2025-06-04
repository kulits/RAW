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


from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import ROTATION_TOKEN_ID, APPEARANCE_TOKEN_ID
from llava.model.language_model.head_utils import get_rotation_head, get_appearance_head, symmetric_orthogonalization, W_ROTATION_MSE, W_APPEARANCE_NORM, W_APPEARANCE_COSINE

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


@dataclass
class FloatsCausalLMOutputWithPast(CausalLMOutputWithPast):
    rotations_pred: Optional[Tuple[torch.FloatTensor]] = None
    appearances_pred: Optional[Tuple[torch.FloatTensor]] = None
    logs: Optional[Dict[str, float]] = None


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotation_head = get_rotation_head(config)
        self.appearance_head = get_appearance_head(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        rotations: Optional[torch.FloatTensor] = None,
        appearances: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, FloatsCausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        logs = {}
        loss = None
        rotations_pred = None
        appearances_pred = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            logs["next_token_loss"] = loss.detach().item()

            rotations_pred_all = self.rotation_head(hidden_states)[:, :-1, :]
            rotations_mask = labels[:, 1:] == ROTATION_TOKEN_ID
            rotations_pred = rotations_pred_all[rotations_mask.to(rotations_pred_all.device)]
            rotations_pred = rotations_pred.reshape(-1, rotations_pred.shape[-1]).float()
            rotations_pred = symmetric_orthogonalization(rotations_pred)

            appearances_pred_all = self.appearance_head(hidden_states)[:, :-1, :]
            appearances_mask = labels[:, 1:] == APPEARANCE_TOKEN_ID
            appearances_pred = appearances_pred_all[appearances_mask.to(appearances_pred_all.device)]
            appearances_pred = appearances_pred.reshape(-1, appearances_pred.shape[-1]).float()

            if rotations is not None:
                rotations = rotations.reshape(*rotations_pred.shape).float()
                rotations = symmetric_orthogonalization(rotations)
                rotation_mse_loss = torch.nn.functional.mse_loss(rotations_pred, rotations)
                loss += rotation_mse_loss * W_ROTATION_MSE
                logs["rotation_mse_loss"] = rotation_mse_loss.detach().item()

            if appearances is not None:
                appearances = appearances.reshape(*appearances_pred.shape).float()
                appearance_norm_loss = (1 - torch.linalg.vector_norm(appearances_pred, dim=-1, ord=2)).square().mean()
                loss += appearance_norm_loss * W_APPEARANCE_NORM
                logs["appearance_norm_loss"] = appearance_norm_loss.detach().item()

                appearance_cosine_loss = 1 - torch.nn.functional.cosine_similarity(appearances_pred, appearances).mean()
                loss += appearance_cosine_loss * W_APPEARANCE_COSINE
                logs["appearance_cosine_loss"] = appearance_cosine_loss.detach().item()

        if not return_dict:
            output = (logits,) + outputs[1:] + (rotations_pred, appearances_pred, logs)
            return (loss,) + output if loss is not None else output

        return FloatsCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rotations_pred=rotations_pred,
            appearances_pred=appearances_pred,
            logs=logs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs_copy = inputs.clone()

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        output_ids = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        if (output_ids == ROTATION_TOKEN_ID).any() or (output_ids == APPEARANCE_TOKEN_ID).any():
            inputs = torch.cat([inputs_copy, output_ids[:, 1:]], dim=1)
            output = self(
                input_ids=inputs,
                labels=inputs,
                attention_mask=None,
                images=images,
                floats=True,
            )
            return output_ids, output.rotations_pred, output.appearances_pred
        return output_ids, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
