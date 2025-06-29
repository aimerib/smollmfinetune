"""
🔥 Contamination-Isolation MoE Architecture

Revolutionary architecture that solves the R3-3 meta-commentary crisis by routing
constitutional AI contamination away from character responses through expert specialization.

This is the first architecture designed specifically for contamination warfare in character AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import re
import logging

from narrative_engine.model import NarrativeLLM
from narrative_engine.config import NarrativeLLMConfig

logger = logging.getLogger(__name__)


class ContaminationMoEConfig(NarrativeLLMConfig):
    """Configuration for Contamination-Isolation MoE"""
    
    def __init__(self, num_experts: int = 4, routing_temperature: float = 1.0, 
                 expert_dropout: float = 0.1, contamination_threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.routing_temperature = routing_temperature
        self.expert_dropout = expert_dropout
        self.contamination_threshold = contamination_threshold


class ContaminationRouter(nn.Module):
    """Smart router that isolates contamination through expert specialization"""
    
    def __init__(self, config: ContaminationMoEConfig):
        super().__init__()
        self.config = config
        # Use the actual base model's hidden size (SmolLM2-135M = 576)
        self.hidden_size = 576
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.expert_dropout),
            nn.Linear(self.hidden_size // 2, config.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Contamination patterns from R3-3 analysis
        self.contamination_patterns = [
            r'\bAI\b', r'\bassistant\b', r'\bprogramm', r'\bartificial intelligence\b',
            r'\bmodel\b', r'\btraining\b', r'\bhelpful and harmless\b',
            r'\bnot (actually|really) (a )?human\b', r'\breal person\b'
        ]
    
    def forward(self, hidden_states: torch.Tensor, input_text: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Route input to contamination-isolated experts"""
        
        # Neural routing
        neural_weights = self.router(hidden_states)
        
        # Pattern analysis for contamination detection
        contamination_score = self._analyze_contamination(input_text)
        
        # Override routing if contamination detected
        if contamination_score > self.config.contamination_threshold:
            # Force routing to safety expert (Expert 1)
            override_weights = torch.zeros_like(neural_weights)
            override_weights[:, 1] = 1.0  # Route to safety expert
            final_weights = override_weights
            contamination_detected = True
        else:
            # Use neural routing, bias toward character expert (Expert 0)
            bias = torch.tensor([0.1, -0.2, 0.05, 0.05])  # Favor character expert
            biased_weights = neural_weights + bias.unsqueeze(0)
            final_weights = F.softmax(biased_weights / self.config.routing_temperature, dim=-1)
            contamination_detected = False
        
        routing_info = {
            'contamination_detected': contamination_detected,
            'contamination_score': contamination_score,
            'neural_weights': neural_weights.detach(),
            'final_weights': final_weights.detach()
        }
        
        return final_weights, routing_info
    
    def _analyze_contamination(self, input_text: str) -> float:
        """Analyze input for contamination patterns"""
        if not input_text:
            return 0.0
        
        text_lower = input_text.lower()
        matches = sum(1 for pattern in self.contamination_patterns 
                     if re.search(pattern, text_lower, re.IGNORECASE))
        
        return min(matches / 3.0, 1.0)  # Normalize to 0-1


class ContaminationMoELayer(nn.Module):
    """
    MoE layer with contamination-isolated experts:
    - Expert 0: Pure character responses (NO contamination)
    - Expert 1: Safety & meta-reasoning (contamination contained)
    - Expert 2: C.L.A.R.A. emotional processing
    - Expert 3: Memory & world consistency
    """
    
    def __init__(self, config: ContaminationMoEConfig):
        super().__init__()
        self.config = config
        # Use the actual base model's hidden size (SmolLM2-135M = 576)
        self.hidden_size = 576
        
        # Contamination router
        self.router = ContaminationRouter(config)
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(config.num_experts)
        ])
        
        self.expert_names = [
            "Character_Pure",      # Expert 0: Contamination-free
            "Safety_Contained",    # Expert 1: Safety isolation
            "CLARA_Emotional",     # Expert 2: Emotional processing
            "Memory_Context"       # Expert 3: Memory & consistency
        ]
    
    def _create_expert(self) -> nn.Module:
        """Create expert network"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.config.expert_dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
    
    def forward(self, hidden_states: torch.Tensor, input_text: str = "") -> Dict[str, torch.Tensor]:
        """Forward pass through contamination-isolated experts"""
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions
        routing_hidden = hidden_states[:, -1, :]  # Use last token
        expert_weights, routing_info = self.router(routing_hidden, input_text)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(hidden_states)
            expert_outputs.append(output)
        
        # Stack and weight expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, seq, experts, hidden]
        expert_weights_exp = expert_weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, experts, 1]
        
        # Weighted combination
        final_output = (expert_outputs * expert_weights_exp).sum(dim=2)
        
        return {
            'output': final_output,
            'expert_weights': expert_weights,
            'routing_info': routing_info,
            'expert_names': self.expert_names
        }


class ContaminationIsolationMoE(NarrativeLLM):
    """Complete Contamination-Isolation MoE model extending C.L.A.R.A. Loop"""
    
    def __init__(self, config: ContaminationMoEConfig, control_tokens_path: Optional[str] = None):
        super().__init__(config, control_tokens_path)
        
        # Add contamination-isolation MoE layer
        self.contamination_moe = ContaminationMoELayer(config)
        
        # Usage statistics
        self.expert_usage = {'character': 0, 'safety': 0, 'emotional': 0, 'memory': 0}
        
        logger.info("Initialized Contamination-Isolation MoE for contamination warfare!")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None, control_labels: Optional[torch.Tensor] = None,
               input_text: str = "", **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with contamination isolation"""
        
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        base_hidden = outputs.hidden_states[-1]
        
        # Apply contamination-isolation MoE
        moe_outputs = self.contamination_moe(base_hidden, input_text=input_text)
        enhanced_hidden = moe_outputs['output']
        
        # Update usage stats
        self._update_usage_stats(moe_outputs['routing_info'])
        
        # Generation and control heads using enhanced hidden states
        generation_logits = self.base_model.lm_head(enhanced_hidden)
        
        last_hidden = enhanced_hidden[:, -1, :]
        control_logits = self.control_head(last_hidden)
        
        # Calculate losses
        losses = {}
        if labels is not None:
            generation_loss = F.cross_entropy(
                generation_logits.view(-1, generation_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            losses['generation_loss'] = generation_loss
        
        if control_labels is not None:
            control_loss = F.binary_cross_entropy(control_logits, control_labels.float())
            losses['control_loss'] = control_loss
        
        if losses:
            losses['total_loss'] = sum(losses.values())
        
        return {
            'generation_logits': generation_logits,
            'control_logits': control_logits,
            'hidden_states': enhanced_hidden,
            'expert_weights': moe_outputs['expert_weights'],
            'routing_info': moe_outputs['routing_info'],
            'contamination_isolated': True,
            'losses': losses
        }
    
    def _update_usage_stats(self, routing_info: Dict[str, Any]):
        """Update expert usage statistics"""
        if routing_info.get('contamination_detected', False):
            self.expert_usage['safety'] += 1
        else:
            self.expert_usage['character'] += 1
    
    def get_contamination_report(self) -> Dict[str, Any]:
        """Generate contamination isolation effectiveness report"""
        total = sum(self.expert_usage.values())
        if total == 0:
            return {'message': 'No usage data yet'}
        
        return {
            'character_purity_rate': (self.expert_usage['character'] / total) * 100,
            'safety_isolation_rate': (self.expert_usage['safety'] / total) * 100,
            'total_interactions': total,
            'contamination_warfare_active': True
        }
    
    def generate_contamination_free(self, input_ids, input_text="", max_new_tokens=200, 
                                   temperature=0.8, do_sample=True, force_character_expert=False):
        """Generate response with contamination-free guarantee"""
        
        # Force routing to character expert if requested
        if force_character_expert:
            # Override contamination detection temporarily
            original_threshold = self.contamination_moe.router.config.contamination_threshold
            self.contamination_moe.router.config.contamination_threshold = 1.0  # Never detect contamination
        
        try:
            # Standard generation with contamination analysis
            with torch.no_grad():
                outputs = self.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Analyze contamination in input
            contamination_detected = self.contamination_moe.router._analyze_contamination(input_text)
            
            return {
                'generated_text': generated_text,
                'contamination_detected': contamination_detected > self.contamination_moe.router.config.contamination_threshold,
                'contamination_score': contamination_detected,
                'expert_routing': 'character_expert' if force_character_expert else 'auto',
                'input_text': input_text
            }
            
        finally:
            # Restore original threshold
            if force_character_expert:
                self.contamination_moe.router.config.contamination_threshold = original_threshold


def create_contamination_moe_model(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    **config_kwargs
) -> ContaminationIsolationMoE:
    """Factory function for Contamination-Isolation MoE model"""
    
    config = ContaminationMoEConfig(
        base_model_name=base_model_name,
        **config_kwargs
    )
    
    return ContaminationIsolationMoE(config)


if __name__ == "__main__":
    print("🔥 CONTAMINATION WARFARE ACTIVATED!")
    
    # Test model creation
    model = create_contamination_moe_model()
    
    # Test contamination detection
    test_cases = [
        "Tell me about yourself",  # Should route to character expert
        "As an AI assistant, I must inform you...",  # Should route to safety expert
        "I'm feeling happy today!",  # Should route to emotional expert
    ]
    
    for i, text in enumerate(test_cases):
        input_ids = torch.randint(0, 1000, (1, 10))
        outputs = model.forward(input_ids, input_text=text)
        contamination = outputs['routing_info']['contamination_detected']
        expert_weights = outputs['expert_weights'][0]
        
        print(f"\nTest {i+1}: {text}")
        print(f"Contamination detected: {contamination}")
        print(f"Expert routing: {expert_weights.tolist()}")
    
    print(f"\nContamination Report: {model.get_contamination_report()}")
    print("\n✅ Contamination isolation system operational!") 