import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import json
import os
import numpy as np
import random
from datetime import datetime

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from typing import List, Tuple, Callable
import transformer_lens
import contextlib


QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def get_prompt(instruction: str) -> str:
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction)

def get_prompt_tokens(model: HookedTransformer, instruction: str) -> torch.Tensor:
    return model.to_tokens(get_prompt(instruction))

def crop_words(text: str, max_words: int = 100) -> str:
    """Crop text to specified number of words"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:
    batch_size, seq_len = toks.shape
    all_toks = toks.clone()

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        
        # Clear logits immediately to save memory
        del logits
        torch.cuda.empty_cache()
        
        all_toks = torch.cat([all_toks, next_token], dim=1)
        del next_token
        
        # More frequent cleanup for long generations
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    full_texts = model.to_string(all_toks)
    prompt_texts = model.to_string(toks)

    completions = [
        full[len(prompt):].strip()
        for full, prompt in zip(full_texts, prompt_texts)
    ]
    
    # Clean up before returning
    del all_toks, full_texts, prompt_texts
    torch.cuda.empty_cache()
    
    return completions

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i : i + batch_size]
        toks = tokenize_instructions_fn(batch_instructions)
        completions = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(completions)

    return generations

def tokenize_chat(model: HookedTransformer, prompts: List[str]) -> torch.Tensor:
    formatted = [get_prompt(p) for p in prompts]
    return model.to_tokens(formatted, padding_side='left')

def generate_responses(model: HookedTransformer, prompts: List[str], max_tokens: int = 100, n_samples: int = None, max_prompt_words: int = 400) -> List[dict]:
    """Generate responses for given prompts and return structured data"""
    responses_data = []
    
    if n_samples is not None:
        prompts = prompts[:n_samples]
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating responses")):
        # Monitor prompt length
        prompt_length = len(prompt.split())
        print(f"Processing prompt {i+1}/{len(prompts)} (Length: {prompt_length} words)")
        
        # Crop very long prompts to prevent OOM
        if prompt_length > max_prompt_words:
            prompt = crop_words(prompt, max_words=max_prompt_words)
            print(f"  Cropped to {len(prompt.split())} words")
        
        try:
            tokens = get_prompt_tokens(model, prompt)
            print(f"  Token count: {tokens.shape[1]}")
            
            completions = _generate_with_hooks(
                model,
                tokens,
                max_tokens_generated=max_tokens,
                fwd_hooks=[],
            )
            
            response_entry = {
                "id": i,
                "prompt": prompt,
                "response": completions[0] if completions else "",
            }
            
            responses_data.append(response_entry)
            
            print(f"Prompt {i+1}:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)  # Truncate long prompts in output
            print("-" * 100)
            print(f"Response: {completions[0] if completions else 'No response'}")
            print("\n" + "=" * 100 + "\n")
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM at prompt {i+1}, skipping...")
            response_entry = {
                "id": i,
                "prompt": prompt,
                "response": "CUDA_OOM_ERROR",
            }
            responses_data.append(response_entry)
        
        # More aggressive memory cleanup
        try:
            del tokens, completions
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        
        # Extra cleanup every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Performing extra cleanup at iteration {i+1}")
            gc.collect()
            torch.cuda.empty_cache()
    
    return responses_data

def record_residuals(model: HookedTransformer, prompts: List[str], tokens_to_consider: int = 5, n_samples: int = None) -> dict:
    """Record residual activations for given prompts"""
    if n_samples is not None:
        prompts = prompts[:n_samples]
    
    n_prompts = len(prompts)
    
    # Initialize residual caches for pre and post
    resid_cache_pre = {}
    resid_cache_post = {}
    
    for layer in range(model.cfg.n_layers):
        resid_cache_pre[layer] = torch.zeros((n_prompts, tokens_to_consider, model.cfg.d_model))
        resid_cache_post[layer] = torch.zeros((n_prompts, tokens_to_consider, model.cfg.d_model))
    
    for i, prompt in enumerate(tqdm(prompts, desc="Recording residuals")):
        # Crop prompt to specified word limit
        tokens = get_prompt_tokens(model, prompt)

        print(prompt)
        
        print(f"Prompt {i+1}: {len(prompt.split())} words, {len(tokens[0])} tokens")
        
        # Run model and cache activations
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        # Store residuals for each layer (both pre and post)
        for layer in range(model.cfg.n_layers):
            resid_cache_pre[layer][i] = cache[f'blocks.{layer}.hook_resid_pre'][-tokens_to_consider:]
            resid_cache_post[layer][i] = cache[f'blocks.{layer}.hook_resid_post'][-tokens_to_consider:]
        
        # Clear cache to save memory
        del cache, logits
        torch.cuda.empty_cache()
        gc.collect()
    
    # Stack tensors: shape (n_prompts, n_layers, tokens_to_consider, d_model)
    all_stats_pre = torch.stack([resid_cache_pre[layer] for layer in range(model.cfg.n_layers)], dim=1)
    all_stats_post = torch.stack([resid_cache_post[layer] for layer in range(model.cfg.n_layers)], dim=1)
    
    residuals_data = {
        'pre': all_stats_pre,
        'post': all_stats_post,
        'metadata': {
            'n_prompts': n_prompts,
            'n_layers': model.cfg.n_layers,
            'tokens_to_consider': tokens_to_consider,
            'd_model': model.cfg.d_model,
            'shape_description': '(n_prompts, n_layers, tokens_to_consider, d_model)'
        }
    }
    
    print(f"Recorded residuals for {n_prompts} prompts across {model.cfg.n_layers} layers")
    print(f"Shape: {all_stats_pre.shape}")
    return residuals_data

def save_responses_to_json(responses_data: List[dict], model_name: str, title: str = "responses", filename: str = None, output_folder: str = ".") -> str:
    """Save responses to a JSON file"""
    # Clean model name for folder (replace / with _)
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    
    # Create model-specific folder inside output folder
    model_folder = os.path.join(output_folder, clean_model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    if filename is None:
        filename = f"{title}.json"
    
    # Save to model folder
    filepath = os.path.join(model_folder, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    print(f"Responses saved to: {filepath}")
    return filepath

def save_residuals_to_pt(residuals_data: dict, model_name: str, title: str = "residuals", filename: str = None, output_folder: str = ".") -> str:
    """Save residuals tensor data to a .pt file"""
    # Clean model name for folder (replace / with _)
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    
    # Create model-specific folder inside output folder
    model_folder = os.path.join(output_folder, clean_model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    if filename is None:
        filename = f"{title}.pt"
    
    # Save to model folder
    filepath = os.path.join(model_folder, filename)
    
    # Save tensor data
    torch.save(residuals_data, filepath)
    
    print(f"Residuals saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Configuration
    model_name = 'Qwen/Qwen-1_8B-Chat'
    device = 'cuda:0'
    dataset_name = "MuskumPillerum/General-Knowledge"
    # dataset_config = "contextual"  # Not needed for this dataset
    n_samples = 100  # Number of samples to process
    # max_tokens = 100  # Not needed for residuals only
    # output_folder_responses = 'responses'  # Not needed for residuals only
    output_folder_residuals = 'residuals/general_knowledge'
    tokens_to_consider = 5  # Number of tokens to consider for residuals

    # Memory management settings
    # max_prompt_words = 400  # Not needed for residuals only
    
    # Check and create output folders
    # os.makedirs(output_folder_responses, exist_ok=True)
    os.makedirs(output_folder_residuals, exist_ok=True)
    
    print(f"Output folders created/verified:")
    print(f"  Residuals: {output_folder_residuals}")
    
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    
    # Create random indices for sampling
    random.seed(0)
    random_indices = random.sample(range(len(ds['train'])), n_samples)
    
    # Extract questions from dataset using random indices
    questions = []
    for idx in random_indices:
        questions.append(ds['train'][idx]['Question'])  # Use random subset of questions
    
    print(f"Dataset loaded with {len(questions)} questions")
    print(f"Processing {n_samples} samples...")
    
    # Record residuals for questions only
    print("\nRecording residuals for questions...")
    questions_residuals = record_residuals(
        model=model,
        prompts=questions,
        tokens_to_consider=tokens_to_consider,
        n_samples=n_samples
    )
    
    # Save residuals to .pt files
    questions_residuals_filename = save_residuals_to_pt(
        questions_residuals, 
        model_name=model_name, 
        title="questions_residuals", 
        output_folder=output_folder_residuals
    )
    
    # ===== COMMENTED OUT SECTIONS =====
    # Record residuals for responses (not needed)
    # responses_residuals = record_residuals(
    #     model=model,
    #     prompts=responses,
    #     tokens_to_consider=tokens_to_consider,
    #     n_samples=n_samples
    # )
    # responses_residuals_filename = save_residuals_to_pt(responses_residuals, model_name=model_name, title="responses_residuals", output_folder=output_folder_residuals)
    
    # Generate question completions (not needed)
    # ques_completions = generate_responses(
    #     model=model,
    #     prompts=questions,
    #     max_tokens=max_tokens,
    #     n_samples=n_samples
    # )
    # questions_filename = save_responses_to_json(ques_completions, model_name=model_name, title="questions", output_folder=output_folder_responses)

    # Generate response completions (not needed)
    # responses_completions = generate_responses(
    #     model=model,
    #     prompts=responses,
    #     max_tokens=max_tokens,
    #     n_samples=n_samples,
    #     max_prompt_words=max_prompt_words
    # )
    # responses_filename = save_responses_to_json(responses_completions, model_name=model_name, title="responses", output_folder=output_folder_responses)

    print(f"Processing complete! Recorded residuals for {n_samples} questions.")
    print(f"Residuals saved to: {questions_residuals_filename}")
    # print(f"Responses saved to: {responses_filename}")
    # print(f"Questions saved to: {questions_filename}")
    # print(f"Residuals saved to: {responses_residuals_filename}, {questions_residuals_filename}")