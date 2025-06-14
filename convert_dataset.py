#!/usr/bin/env python3
"""
Convert roleplay datasets from current format to improved special token format.

Current format:  Prompt: <question> || Response: <answer>
New format:      <|prompt|> <question>
                 <|tag|> <answer>
                 <|endofresponse|>
"""

import os
import json
import re
from typing import Dict, List, Tuple

def parse_current_format(raw_data: str) -> List[Tuple[str, str]]:
    """Parse current 'Prompt: ... || Response: ...' format."""
    pairs = []
    
    # Split by newlines and process each line
    lines = raw_data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for the pattern "Prompt: ... || Response: ..."
        if ' || ' in line and line.startswith('Prompt: '):
            try:
                prompt_part, response_part = line.split(' || ', 1)
                
                # Extract prompt (remove "Prompt: " prefix)
                if prompt_part.startswith('Prompt: '):
                    prompt = prompt_part[8:].strip()  # Remove "Prompt: "
                else:
                    continue
                    
                # Extract response (remove "Response: " prefix)
                if response_part.startswith('Response: '):
                    response = response_part[9:].strip()  # Remove "Response: "
                else:
                    continue
                    
                pairs.append((prompt, response))
                
            except ValueError:
                # Skip malformed lines
                continue
                
    return pairs

def convert_to_new_format(pairs: List[Tuple[str, str]], tag: str) -> str:
    """Convert parsed pairs to new special token format."""
    formatted_lines = []
    
    for prompt, response in pairs:
        formatted_lines.extend([
            f"<|prompt|> {prompt}",
            f"<|{tag}|> {response}",
            "<|endofresponse|>",
            ""  # Empty line for separation
        ])
    
    return "\n".join(formatted_lines)

def convert_to_jsonl(pairs: List[Tuple[str, str]], tag: str) -> List[str]:
    """Convert pairs to JSONL format."""
    jsonl_lines = []
    
    for prompt, response in pairs:
        entry = {
            "prompt": prompt,
            "response": response,
            "tag": tag
        }
        jsonl_lines.append(json.dumps(entry))
    
    return jsonl_lines

def main():
    # Define the datasets (complete data from your notebook)
    datasets = {}
    

    from dataset_definitions import DATASETS
    datasets = DATASETS    
    # Create output directories
    os.makedirs("data/formatted", exist_ok=True)
    os.makedirs("data/jsonl", exist_ok=True)
    
    print("Converting datasets...")
    
    # Convert each dataset
    for tag, raw_data in datasets.items():
        print(f"Processing {tag}...")
        
        # Parse the current format
        pairs = parse_current_format(raw_data)
        print(f"  Found {len(pairs)} prompt-response pairs")
        
        # Convert to new special token format
        formatted_text = convert_to_new_format(pairs, tag)
        
        # Save formatted text
        with open(f"data/formatted/{tag}.txt", "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        # Convert to JSONL and save
        jsonl_lines = convert_to_jsonl(pairs, tag)
        with open(f"data/jsonl/{tag}.jsonl", "w", encoding="utf-8") as f:
            for line in jsonl_lines:
                f.write(line + "\n")
        
        print(f"  Saved to data/formatted/{tag}.txt and data/jsonl/{tag}.jsonl")
    
    # Create a combined formatted file
    print("\nCreating combined dataset...")
    combined_lines = []
    for tag, raw_data in datasets.items():
        pairs = parse_current_format(raw_data)
        formatted = convert_to_new_format(pairs, tag)
        combined_lines.append(formatted)
    
    combined_text = "\n".join(combined_lines)
    with open("data/formatted/combined.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)
    
    print("✓ Conversion complete!")
    print("\nFiles created:")
    print("  data/formatted/ - Text files with special token format")
    print("  data/jsonl/ - JSONL files for alternative loading")
    print("  data/formatted/combined.txt - All datasets combined")

if __name__ == "__main__":
    main() 