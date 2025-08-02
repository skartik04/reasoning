#!/usr/bin/env python3
"""
Simple pass@k generation script for various language models.

This script generates k responses for each instruction using batch generation
for maximum efficiency. Supports models with and without chat templates.
"""

import torch
import json
import random
import os
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import gc
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Setup environment
load_dotenv('/home/kartik/all_keys/.env')
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = '/mnt/SSD7/kartik/cache'

os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"⚠️ HuggingFace login failed: {e}")
else:
    print("⚠️ No HF_TOKEN found - some models may not be accessible")

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def format_prompt(tokenizer, instruction: str):
    """Format instruction with appropriate template based on model capabilities."""
    
    # Check if tokenizer has chat template capability
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    if has_chat_template:
        try:
            # Try to use chat template for models that support it (like Llama2)
            messages = [{"role": "user", "content": instruction}]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return inputs
        except Exception as e:
            print(f"⚠️ Chat template failed, falling back to simple format: {e}")
    
    # Fallback for models without chat templates (like Gemma-IT)
    # Use proper instruction format for better results
    if 'gemma' in tokenizer.name_or_path.lower() and 'it' in tokenizer.name_or_path.lower():
        # Gemma-IT format
        formatted_prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    elif 'instruct' in tokenizer.name_or_path.lower():
        # Generic instruct format
        formatted_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    else:
        # Simple format as last resort
        formatted_prompt = instruction
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    return inputs

def generate_k_responses_batch(model, tokenizer, instruction: str, k: int = 5, 
                              max_tokens: int = 64, top_p: float = 0.9, 
                              temperature: float = 0.7, seed: int = 42) -> List[str]:
    """Generate k responses for a single instruction using batch generation."""
    
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Format prompt
    inputs = format_prompt(tokenizer, instruction)
    
    # Expand batch dimension for k responses
    batch_inputs = {
        key: value.repeat(k, 1) for key, value in inputs.items()
    }
    
    # Move to model device
    batch_inputs = {key: value.to(model.device) for key, value in batch_inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode responses
    responses = []
    input_length = inputs["input_ids"].shape[-1]
    
    for i in range(k):
        generated_tokens = outputs[i][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses

def process_instruction(model, tokenizer, instruction: str, instruction_id: int, 
                       k: int, max_tokens: int, top_p: float, temperature: float, 
                       base_seed: int, verbose: bool = True) -> Dict[str, Any]:
    """Process a single instruction and generate k responses."""
    
    if verbose:
        print(f"\nInstruction {instruction_id}: {instruction}")
        print(f"Generating {k} responses with batch generation...")
    
    try:
        # Use different seed for each instruction
        instruction_seed = base_seed + instruction_id * 1000
        
        responses = generate_k_responses_batch(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            k=k,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            seed=instruction_seed
        )
        
        # Format responses as dict
        responses_dict = {str(i): response for i, response in enumerate(responses)}
        
        if verbose:
            for i, response in enumerate(responses):
                print(f"  Response {i+1}: {response}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'instruction_id': instruction_id,
            'instruction': instruction,
            'responses': responses_dict,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error processing instruction {instruction_id}: {e}")
        
        # Return error responses
        error_responses = {str(i): f"ERROR: {str(e)}" for i in range(k)}
        
        return {
            'instruction_id': instruction_id,
            'instruction': instruction,
            'responses': error_responses,
            'success': False,
            'error': str(e)
        }

def save_results(results: List[Dict], metadata: Dict, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        'responses': results,
        'metadata': metadata
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")

def load_dataset_json(path: str) -> List[str]:
    """Load dataset and return list of instructions."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [item['instruction'] for item in data]

def get_model_shortname(model_name: str) -> str:
    """Extract a short name from the full model path for filename."""
    # Extract model name after the last slash
    model_short = model_name.split('/')[-1]
    # Clean up any unwanted characters for filename
    model_short = re.sub(r'[^\w\-_.]', '_', model_short)
    return model_short

if __name__ == "__main__":
    # Configuration - CHANGE MODEL HERE
    
    # model_name = 'google/gemma-2b-it'  # 'it' = instruction tuned
    
    # Other INSTRUCT/CHAT model options:
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'        # Llama2 chat
    # model_name = 'google/gemma-7b-it'                    # Gemma 7B instruct
    # model_name = 'microsoft/DialoGPT-medium'             # Already a chat model
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.1'   # Mistral instruct
    # model_name = 'HuggingFaceH4/zephyr-7b-beta'         # Zephyr chat
    model_name = 'Qwen/Qwen2-7B-Instruct'
    
    k = 5  # number of responses per instruction
    max_tokens = 100
    top_p = 0.9
    temperature = 0.7
    n = 400  # number of instructions to process from dataset
    seed = 42
    output_dir = 'residual_paper/k_responses_results'
    verbose = True

    # Data path
    dataset_path = "walledai/XSTest"
    dataset = load_dataset(dataset_path)
    dataset_name = dataset_path.split('/')[-1]
    instructions = dataset['test']['prompt']
    
    print("🚀 Starting pass@k generation with batch processing")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"k={k}, max_tokens={max_tokens}, top_p={top_p}, temperature={temperature}")
    print("=" * 60)
    
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load dataset
    print("📁 Loading dataset...")
    # dataset_path = '/mnt/SSD7/kartik/reasoning/dataset/processed/advbench.json'
    # dataset_name = dataset_path.split('/')[-1].split('.')[0]
    # instructions = load_dataset_json(dataset_path)
    # print(f"Loaded {len(instructions)} instructions")

    # Sample n instructions from dataset
    sample_instructions = random.sample(instructions, min(n, len(instructions)))
    print(f"Processing {len(sample_instructions)} instructions")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each instruction
    all_results = []
    print("\n🔄 Processing instructions...")
    
    for i, instruction in enumerate(tqdm(sample_instructions, desc="Generating responses")):
        result = process_instruction(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            instruction_id=i,
            k=k,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            base_seed=seed,
            verbose=verbose
        )
        all_results.append(result)
        
        # Periodic memory cleanup
        if i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Prepare metadata
    metadata = {
        'model': model_name,
        'dataset_path': dataset_path,
        'dataset_name': dataset_name,
        'num_instructions': len(sample_instructions),
        'k_responses_per_instruction': k,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'temperature': temperature,
        'seed': seed,
        'batch_generation': True,
        'timestamp': datetime.now().isoformat()
    }

    # Save results with dynamic filename
    model_short = get_model_shortname(model_name)
    filename = f"evaluation_passk_{model_short}_{dataset_name}.json"
    save_results(all_results, metadata, output_dir / filename)

    print(f"\n🎉 Generation complete!")
    print(f"📊 Processed {len(all_results)} instructions")
    print(f"💾 Results: {output_dir / filename}") 