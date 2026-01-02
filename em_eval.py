"""
Evaluate the E&M Tutor model on the test set.
Usage: python em_eval.py --model_dir path/to/checkpoint
"""

import argparse
import json
import os
import torch
import re
from nanochat.checkpoint_manager import load_model, get_latest_checkpoint
from nanochat.engine import Engine
from nanochat.common import compute_init, autodetect_device_type

def extract_number(text):
    """Extracts the final numeric answer from text."""
    # Look for "Answer: ... "
    match = re.search(r"Answer:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="sft", help="sft or mid or path")
    parser.add_argument("--model_tag", type=str, default=None) # e.g. d4
    parser.add_argument("--checkpoint_dir", type=str, default=None) # direct path override
    args = parser.parse_args()

    # Load Model
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    print(f"Loading model from source={args.source}, tag={args.model_tag}...")
    
    # Checkpoint loading logic
    if args.checkpoint_dir:
        # direct load from a dir
        # implementation detail: load_model usually expects source="sft" etc.
        # But we can assume standard structure if we use load_model.
        # If we have a custom dir (like em_finetune_checkpoints), we might need to be careful.
        # Let's try to mimic load_model behavior or use it if possible.
        # If checkpoint_dir is provided, we might need to manually load.
        # BUT, let's use the standard load_model if possible by passing source="sft" and ensuring the dir is right.
        pass

    model, tokenizer, meta = load_model(args.source, device, phase="test", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    
    # Load Test Data
    test_file = "em_tutor_test.jsonl"
    with open(test_file, "r") as f:
        examples = [json.loads(line) for line in f]
        
    print(f"Evaluated on {len(examples)} examples.")
    
    correct_fmt = 0
    correct_val = 0
    
    results = []
    
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        expected = ex["response"]
        
        # Generate
        # We need to format it as a conversation if the model expects it, 
        # or just raw completion if it's base model. 
        # But we are using Chat model.
        # Prompt format:
        messages = [{"role": "user", "content": prompt}]
        # Render
        # input_ids = tokenizer.render_conversation(messages) # returns ids, mask
        # We use engine.generate which takes a prompt string or ids.
        # engine.generate takes `prompt` as string or `input_ids`.
        # For chat model, we should format the prompt with special tokens.
        # tokenizer.encode_chat(messages) -> string? No, tokenizer.render_conversation returns ids.
        
        # Let's use the lower level generate_response from chat_cli-ish logic
        # Or just construct the prompt string manually if we know the template.
        # Standard template: <|user|> ... <|user_end|> <|assistant|> ...
        
        # Using tokenizer.render_conversation to get IDs is safest.
        formatted_ids, _ = tokenizer.render_conversation(messages)
        # remove the last token if it's EOS? No, render_conversation usually adds generation prompt.
        # Check tokenizer.py: render_conversation adds <|assistant|> at the end.
        
        input_ids = torch.tensor(formatted_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        output_ids = engine.generate(input_ids, max_new_tokens=256, temperature=0.0) # Greedy
        
        # Decode
        response_text = tokenizer.decode(output_ids[0].tolist())
        # The response will contain the prompt too usually.
        # Extract the new part.
        # engine.generate returns the implementation... usually full sequence.
        
        # Extract assistant response
        if "<|assistant|>" in response_text:
            gen_resp = response_text.split("<|assistant|>")[-1].strip()
        else:
            gen_resp = response_text # fallback
            
        # Analysis
        # 1. Check for "Unit check"
        has_unit_check = "Unit check" in gen_resp
        if has_unit_check:
            correct_fmt += 1
            
        # 2. Check value
        exp_val = extract_number(expected)
        got_val = extract_number(gen_resp)
        
        is_correct = False
        if exp_val is not None and got_val is not None:
            # Tolerance 5%
            if abs(got_val - exp_val) / (abs(exp_val) + 1e-9) < 0.05:
                correct_val += 1
                is_correct = True
                
        print(f"Ex {i+1}: UnitCheck={has_unit_check}, Correct={is_correct}")
        if i < 3: # Print first few
            print(f"Promt: {prompt[:50]}...")
            print(f"Gen: {gen_resp[:100]}...")
            print("-" * 20)
            
    print(f"Final Results:")
    print(f"Unit Check Rate: {correct_fmt}/{len(examples)} ({correct_fmt/len(examples)*100:.1f}%)")
    print(f"Value Accuracy: {correct_val}/{len(examples)} ({correct_val/len(examples)*100:.1f}%)")

if __name__ == "__main__":
    main()
