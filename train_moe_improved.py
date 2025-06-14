#!/usr/bin/env python3
"""
Improved MoE training script with proper special token handling.

This script:
- Adds special tokens to the tokenizer 
- Resizes model embeddings accordingly
- Uses the new special token dataset format
- Provides cleaner routing signals for the MoE gate
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors.torch import save_file, load_file

# Special tokens for our improved format
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<|prompt|>",
        "<|backstory|>", 
        "<|personality|>",
        "<|speech|>",
        "<|nsfw|>",
        "<|endofresponse|>"
    ]
}

class ImprovedRoleplayDataset(Dataset):
    """Dataset for the new special token format."""
    
    def __init__(self, formatted_text: str, tokenizer: AutoTokenizer, max_len: int = 256):
        self.lines = [ln.strip() for ln in formatted_text.split("\n") if ln.strip()]
        self.tok = tokenizer
        self.max_len = max_len
        
        # Filter out just the text lines (not empty separator lines)
        self.lines = [line for line in self.lines if line]
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        
        # Tokenize with our special tokens
        encoded = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        return {k: v.squeeze(0) for k, v in encoded.items()}

def prepare_tokenizer_and_model(base_model_name: str, device: str = "cpu"):
    """Load tokenizer and model, add special tokens, and resize embeddings."""
    
    print(f"Loading tokenizer and model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add our special tokens
    print("Adding special tokens...")
    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(f"Added {num_added} special tokens to tokenizer")
    
    # Load model and resize embeddings
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if num_added > 0:
        print("Resizing model embeddings...")
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    return tokenizer, model

def load_formatted_datasets(data_dir: str = "data/formatted") -> Dict[str, str]:
    """Load the converted datasets in special token format."""
    
    datasets = {}
    
    for filename in ["backstory.txt", "personality.txt", "speech.txt", "nsfw.txt"]:
        tag = filename.split(".")[0]
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                datasets[tag] = f.read()
            print(f"Loaded {tag} dataset from {filepath}")
        else:
            print(f"Warning: {filepath} not found, skipping {tag}")
    
    return datasets

class RoleplayMoE(nn.Module):
    """Improved MoE with better special token handling."""
    
    def __init__(self, experts: List[nn.Module]):
        super().__init__()
        if not experts:
            raise ValueError("Must pass >=1 experts")
        
        self.experts = nn.ModuleList(experts)
        d_model = experts[0].config.hidden_size
        
        # Gate network - takes embeddings and outputs expert weights
        self.gate = nn.Sequential(
            nn.Linear(d_model, len(experts)),
            nn.Softmax(dim=-1)
        )
        
        # Freeze expert parameters initially (only train gate)
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get embeddings from first expert (they all share the same embedding layer structure)
        with torch.no_grad():
            embeddings = self.experts[0].get_input_embeddings()(input_ids)
        
        # Gate computation: average embeddings across sequence length for each sample
        # This gives us a representation of the whole input for routing
        gate_input = embeddings.mean(dim=1)  # [batch_size, hidden_size]
        expert_weights = self.gate(gate_input)  # [batch_size, num_experts]
        
        # Get outputs from each expert
        logits = None
        for i, expert in enumerate(self.experts):
            expert_logits = expert(input_ids, attention_mask=attention_mask).logits
            
            # Weight this expert's output by the gate
            weight = expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            weighted_logits = expert_logits * weight
            
            if logits is None:
                logits = weighted_logits
            else:
                logits = logits + weighted_logits
        
        return CausalLMOutputWithPast(logits=logits)
    
    def get_input_embeddings(self):
        return self.experts[0].get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.experts[0].get_output_embeddings()

def fine_tune_expert(
    base_model_name: str,
    formatted_data: str,
    tokenizer: AutoTokenizer,
    epochs: int = 10,
    lr: float = 5e-5,
    batch_size: int = 8,
    device: str = "cpu"
) -> nn.Module:
    """Fine-tune a single expert on formatted data."""
    
    # Create fresh model instance for this expert
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Resize embeddings to match tokenizer (with special tokens)
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    # Create dataset and dataloader
    dataset = ImprovedRoleplayDataset(formatted_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for causal language modeling
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Flatten for loss computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            loss = criterion(flat_logits, flat_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    model.eval()
    return model

def train_moe_gate(
    moe: RoleplayMoE,
    combined_data: str,
    tokenizer: AutoTokenizer,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cpu"
):
    """Train the MoE gating network on combined data."""
    
    moe.to(device)
    
    # Create dataset from combined formatted data
    dataset = ImprovedRoleplayDataset(combined_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Only optimize gate parameters
    optimizer = torch.optim.AdamW(moe.gate.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(epochs):
        moe.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward through MoE
            outputs = moe(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for causal language modeling
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Flatten for loss computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            loss = criterion(flat_logits, flat_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe.gate.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  MoE Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

# HuggingFace wrapper classes (same as before but updated for new format)
class RoleplayMoEConfig(PretrainedConfig):
    model_type = "roleplay_moe"
    
    def __init__(self, base_model_name: str = "", num_experts: int = 1, **kwargs):
        self.base_model_name = base_model_name
        self.num_experts = num_experts
        self.auto_map = {
            "AutoConfig": "model.RoleplayMoEConfig",
            "AutoModelForCausalLM": "model.RoleplayMoEForCausalLM",
        }
        super().__init__(**kwargs)

class RoleplayMoEForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = RoleplayMoEConfig
    
    def __init__(self, config: RoleplayMoEConfig):
        super().__init__(config)
        
        # Create experts - we'll load the weights separately
        experts = []
        for _ in range(config.num_experts):
            expert = AutoModelForCausalLM.from_pretrained(config.base_model_name)
            experts.append(expert)
        
        self.moe = RoleplayMoE(experts)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.moe(input_ids, attention_mask=attention_mask)
    
    def get_input_embeddings(self):
        return self.moe.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.moe.get_output_embeddings()
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask")
        }

def save_moe_model(moe: RoleplayMoE, save_path: str, base_model_name: str, tokenizer: AutoTokenizer):
    """Save the trained MoE model in HuggingFace format."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create wrapper for saving
    config = RoleplayMoEConfig(
        base_model_name=base_model_name,
        num_experts=len(moe.experts)
    )
    
    wrapper = RoleplayMoEForCausalLM(config)
    wrapper.moe = moe
    
    # Save state dict
    state_dict = {k: v.cpu() for k, v in wrapper.state_dict().items()}
    save_file(state_dict, os.path.join(save_path, "model.safetensors"))
    
    # Save config
    config.save_pretrained(save_path)
    
    # Save updated tokenizer (with special tokens)
    tokenizer.save_pretrained(save_path)
    
    # Create model.py file for loading
    write_model_file(save_path)
    
    print(f"✓ MoE model saved to {save_path}")

def write_model_file(save_dir: str):
    """Write the model.py file needed for HuggingFace auto-loading."""
    
    model_code = '''import torch, torch.nn as nn
from typing import List
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RoleplayMoE(nn.Module):
    def __init__(self, experts: List[nn.Module]):
        super().__init__()
        assert experts
        self.experts = nn.ModuleList(experts)
        d = experts[0].config.hidden_size
        self.gate = nn.Sequential(nn.Linear(d, len(experts)), nn.Softmax(dim=-1))
        for p in self.experts.parameters(): 
            p.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():
            embeddings = self.experts[0].get_input_embeddings()(input_ids)
        gate_input = embeddings.mean(dim=1)
        expert_weights = self.gate(gate_input)
        
        logits = None
        for i, expert in enumerate(self.experts):
            expert_logits = expert(input_ids, attention_mask=attention_mask).logits
            weight = expert_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            weighted = expert_logits * weight
            logits = weighted if logits is None else logits + weighted
        
        return CausalLMOutputWithPast(logits=logits)
    
    def get_input_embeddings(self): 
        return self.experts[0].get_input_embeddings()
    
    def get_output_embeddings(self): 
        return self.experts[0].get_output_embeddings()

class RoleplayMoEConfig(PretrainedConfig):
    model_type = 'roleplay_moe'
    
    def __init__(self, base_model_name='', num_experts=1, **kwargs):
        self.base_model_name = base_model_name
        self.num_experts = num_experts
        self.auto_map = {
            'AutoConfig': 'model.RoleplayMoEConfig',
            'AutoModelForCausalLM': 'model.RoleplayMoEForCausalLM',
        }
        super().__init__(**kwargs)

class RoleplayMoEForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = RoleplayMoEConfig
    
    def __init__(self, config):
        super().__init__(config)
        experts = [AutoModelForCausalLM.from_pretrained(config.base_model_name) 
                  for _ in range(config.num_experts)]
        self.moe = RoleplayMoE(experts)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.moe(input_ids, attention_mask=attention_mask)
    
    def get_input_embeddings(self): 
        return self.moe.get_input_embeddings()
    
    def get_output_embeddings(self): 
        return self.moe.get_output_embeddings()
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask")
        }
'''
    
    with open(os.path.join(save_dir, "model.py"), "w") as f:
        f.write(model_code)

def main():
    """Main training function."""
    
    # Configuration
    base_model = "HuggingFaceTB/SmolLM2-135M"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print("="*50)
    
    # Step 1: Prepare tokenizer with special tokens
    print("Step 1: Preparing tokenizer and base model...")
    tokenizer, _ = prepare_tokenizer_and_model(base_model, device)
    
    # Step 2: Load formatted datasets
    print("\nStep 2: Loading formatted datasets...")
    datasets = load_formatted_datasets()
    
    if not datasets:
        print("Error: No datasets found. Please run convert_dataset.py first.")
        return
    
    # Step 3: Train individual experts
    print("\nStep 3: Training individual experts...")
    experts = []
    
    for tag, data in datasets.items():
        print(f"\nTraining {tag} expert...")
        expert = fine_tune_expert(
            base_model, data, tokenizer,
            epochs=10, lr=5e-5, batch_size=8, device=device
        )
        experts.append(expert)
    
    # Step 4: Create MoE and train gate
    print("\nStep 4: Training MoE gate network...")
    moe = RoleplayMoE(experts)
    
    # Combine all formatted data for gate training
    combined_data = "\n\n".join(datasets.values())
    
    train_moe_gate(
        moe, combined_data, tokenizer,
        epochs=10, lr=1e-4, batch_size=8, device=device
    )
    
    # Step 5: Save the model
    print("\nStep 5: Saving MoE model...")
    save_moe_model(moe, "./moe_roleplay_improved", base_model, tokenizer)
    
    print("\n" + "="*50)
    print("✓ Training complete!")
    print(f"Model saved to: ./moe_roleplay_improved")
    print("\nTo use the model:")
    print("from transformers import AutoTokenizer, AutoModelForCausalLM")
    print("tokenizer = AutoTokenizer.from_pretrained('./moe_roleplay_improved')")  
    print("model = AutoModelForCausalLM.from_pretrained('./moe_roleplay_improved', trust_remote_code=True)")

if __name__ == "__main__":
    main() 