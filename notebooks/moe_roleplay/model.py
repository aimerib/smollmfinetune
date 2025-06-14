import torch, torch.nn as nn
from typing import List
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RoleplayMoE(nn.Module):
    def __init__(self, experts: List[nn.Module]):
        super().__init__(); assert experts
        self.experts = nn.ModuleList(experts)
        d = experts[0].config.hidden_size
        self.gate = nn.Sequential(nn.Linear(d, len(experts)), nn.Softmax(dim=-1))
        for p in self.experts.parameters(): p.requires_grad = False
    def forward(self, input_ids, attention_mask=None, **_):
        with torch.no_grad():
            embeds = self.experts[0].get_input_embeddings()(input_ids)
        w = self.gate(embeds.mean(dim=1))
        logits = None
        for i, exp in enumerate(self.experts):
            l = exp(input_ids, attention_mask=attention_mask).logits
            weighted = l * w[:, i].unsqueeze(-1).unsqueeze(-1)
            logits = weighted if logits is None else logits + weighted
        return CausalLMOutputWithPast(logits=logits)
    def get_input_embeddings(self): return self.experts[0].get_input_embeddings()
    def get_output_embeddings(self): return self.experts[0].get_output_embeddings()

class RoleplayMoEConfig(PretrainedConfig):
    model_type = 'roleplay_moe'
    def __init__(self, base_model_name='', num_experts=1, **kw):
        self.base_model_name = base_model_name; self.num_experts = num_experts
        self.auto_map = {
            'AutoConfig': 'model.RoleplayMoEConfig',
            'AutoModelForCausalLM': 'model.RoleplayMoEForCausalLM',
        }; super().__init__(**kw)

class RoleplayMoEForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = RoleplayMoEConfig
    def __init__(self, config):
        super().__init__(config)
        experts=[AutoModelForCausalLM.from_pretrained(config.base_model_name)
                 for _ in range(config.num_experts)]
        self.moe = RoleplayMoE(experts)
    def forward(self, input_ids, attention_mask=None, **kw):
        return self.moe(input_ids, attention_mask=attention_mask)
    def get_input_embeddings(self): return self.moe.get_input_embeddings()
    def get_output_embeddings(self): return self.moe.get_output_embeddings()
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
         return {
             "input_ids": input_ids,
             "attention_mask": kwargs.get("attention_mask")
        }
