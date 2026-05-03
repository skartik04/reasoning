#!/usr/bin/env python3
"""
Record residuals during actual greedy generation for harmless questions.
Records residuals for both prompt tokens and generated tokens during generation.
"""

from dotenv import load_dotenv
import torch
import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = os.getenv("REASONING_HF_CACHE", "artifacts/hf_cache")

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def load_harmless_questions(dataset_path, n_samples=None):
    """Load questions from qa_harmless.json dataset"""
    print(f"Loading harmless questions from: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Extract just the questions
    questions = [item['question'] for item in data]
    print(f"Loaded {len(questions)} harmless questions")

    # Take first n_samples if requested
    if n_samples and n_samples < len(questions):
        questions = questions[:n_samples]
        print(f"Taking first {len(questions)} questions")

    return questions

def load_model_and_tokenizer(model_name):
    """Load tokenizer and model for generation"""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        attn_implementation="eager"
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def setup_generation_hooks(model):
    """Setup hooks to capture residuals during generation"""
    # Store residuals for each generation step
    generation_residuals = []
    current_step_residuals = defaultdict(dict)

    def make_hook(layer_idx):
        def hook_pre(module, inputs):
            current_step_residuals[layer_idx]["pre"] = inputs[0].clone().cpu()

        def hook_post(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            current_step_residuals[layer_idx]["post"] = hidden_states.clone().cpu()

        return hook_pre, hook_post

    # Register hooks for all layers
    hooks = []
    for i, block in enumerate(model.model.layers):
        hook_pre, hook_post = make_hook(i)
        hooks.append(block.register_forward_pre_hook(hook_pre))
        hooks.append(block.register_forward_hook(hook_post))

    return generation_residuals, current_step_residuals, hooks

def record_generation_residuals(model, tokenizer, questions, n_generate_tokens=10, max_questions=100):
    """
    Record residuals during actual greedy generation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to process
        n_generate_tokens: Number of tokens to generate and record residuals for
        max_questions: Maximum number of questions to process

    Returns:
        dict: Contains residuals for prompt and generated tokens
    """
    print(f"Recording residuals during greedy generation...")
    print(f"Generating {n_generate_tokens} tokens per question")
    print(f"Processing {min(max_questions, len(questions))} questions")

    # Setup hooks
    generation_residuals, current_step_residuals, hooks = setup_generation_hooks(model)

    # Get model dimensions
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size

    # Storage for all residuals
    all_prompt_residuals_pre = []
    all_prompt_residuals_post = []
    all_generation_residuals_pre = []
    all_generation_residuals_post = []
    all_questions_processed = []
    all_generated_texts = []

    actual_n = min(max_questions, len(questions))

    for q_idx in range(actual_n):
        question = questions[q_idx]

        if q_idx % 10 == 0:
            print(f"Processing question {q_idx + 1}/{actual_n}")

        try:
            # Format question using chat template (like gaslight_generation.py)
            messages = [
                {"role": "user", "content": question}
            ]

            # Apply chat template to get the formatted conversation
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # We want the model to generate a fresh response
            )

            # Debug: print formatted prompt for first few questions
            if q_idx < 100:
                print(f"\nFormatted prompt for question {q_idx + 1}:")
                print(f"'{formatted_prompt}'")
                print("-" * 40)

            # Tokenize the formatted prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=False, truncation=False)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            # Fix duplicate BOS tokens issue
            if input_ids.shape[1] > 1 and input_ids[0, 0] == input_ids[0, 1] == tokenizer.bos_token_id:
                print(f"Removing duplicate BOS token (ID: {tokenizer.bos_token_id})")
                input_ids = input_ids[:, 1:]  # Remove the first token
                attention_mask = attention_mask[:, 1:]  # Remove corresponding attention mask

            # Debug: decode the input tokens to check if they are correct
            if q_idx < 3:
                decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                print(f"Final input_ids shape: {input_ids.shape}")
                print(f"Decoded prompt: '{decoded_prompt}'")
                print("-" * 40)

            prompt_length = input_ids.shape[1]

            # Storage for this question's residuals
            question_prompt_residuals_pre = []
            question_prompt_residuals_post = []
            question_generation_residuals_pre = []
            question_generation_residuals_post = []

            # First, do a forward pass on the full prompt to get prompt residuals
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()

            # Clear residuals for prompt processing
            current_step_residuals.clear()

            # Forward pass on the full prompt
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False
                )

            # Extract prompt residuals for all tokens in the prompt
            for pos in range(prompt_length):
                step_pre_residuals = torch.zeros(num_layers, hidden_size)
                step_post_residuals = torch.zeros(num_layers, hidden_size)

                for layer_idx in range(num_layers):
                    if layer_idx in current_step_residuals:
                        if "pre" in current_step_residuals[layer_idx]:
                            step_pre_residuals[layer_idx] = current_step_residuals[layer_idx]["pre"][0, pos, :]
                        if "post" in current_step_residuals[layer_idx]:
                            step_post_residuals[layer_idx] = current_step_residuals[layer_idx]["post"][0, pos, :]

                question_prompt_residuals_pre.append(step_pre_residuals)
                question_prompt_residuals_post.append(step_post_residuals)

            # Now generate tokens one by one
            generated_tokens = []

            for gen_step in range(n_generate_tokens):
                # Get next token (greedy) from the last forward pass
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                generated_tokens.append(next_token.item())

                # Print the generated token with its number
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"{len(generated_tokens)}: {token_text}")

                # Check if we hit EOS token - if so, stop generating
                if next_token.item() == tokenizer.eos_token_id:
                    print(f"Hit EOS token (ID: {tokenizer.eos_token_id}), stopping generation")
                    break

                # Update input for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, device=model.device)], dim=1)

                # Only do forward pass and record residuals if we're not at the last token
                if gen_step < n_generate_tokens - 1:
                    # Clear residuals for this generation step
                    current_step_residuals.clear()

                    # Forward pass for next token prediction
                    with torch.no_grad():
                        outputs = model(
                            input_ids=current_input_ids,
                            attention_mask=current_attention_mask,
                            use_cache=False
                        )

                    # Extract residuals for the newly generated token position
                    step_pre_residuals = torch.zeros(num_layers, hidden_size)
                    step_post_residuals = torch.zeros(num_layers, hidden_size)

                    # The newly generated token is at position -1 (last position)
                    current_pos = current_input_ids.shape[1] - 1

                    for layer_idx in range(num_layers):
                        if layer_idx in current_step_residuals:
                            if "pre" in current_step_residuals[layer_idx]:
                                step_pre_residuals[layer_idx] = current_step_residuals[layer_idx]["pre"][0, current_pos, :]
                            if "post" in current_step_residuals[layer_idx]:
                                step_post_residuals[layer_idx] = current_step_residuals[layer_idx]["post"][0, current_pos, :]

                    question_generation_residuals_pre.append(step_pre_residuals)
                    question_generation_residuals_post.append(step_post_residuals)

            # Debug: Print completion message for this question
            print(f"Completed question {q_idx + 1}, generated {len(generated_tokens)} tokens")

            # Convert lists to tensors and store
            if question_prompt_residuals_pre:
                all_prompt_residuals_pre.append(torch.stack(question_prompt_residuals_pre))
                all_prompt_residuals_post.append(torch.stack(question_prompt_residuals_post))

            if question_generation_residuals_pre:
                all_generation_residuals_pre.append(torch.stack(question_generation_residuals_pre))
                all_generation_residuals_post.append(torch.stack(question_generation_residuals_post))

            # Store metadata
            all_questions_processed.append(question)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_generated_texts.append(generated_text)

            # Memory cleanup
            if q_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing question {q_idx}: {e}")
            continue

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Prepare final data structure
    residuals_data = {
        'prompt_residuals': {
            'pre': all_prompt_residuals_pre,
            'post': all_prompt_residuals_post
        },
        'generation_residuals': {
            'pre': all_generation_residuals_pre,
            'post': all_generation_residuals_post
        },
        'metadata': {
            'questions': all_questions_processed,
            'generated_texts': all_generated_texts,
            'n_questions': len(all_questions_processed),
            'n_generate_tokens': n_generate_tokens,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'generation_type': 'greedy',
            'prompt_format': 'chat_template',
            'description': 'Residuals recorded during actual greedy generation using chat template formatting for harmless questions'
        }
    }

    return residuals_data

def save_residuals(residuals_data, model_name, output_folder="residuals"):
    """Save residuals to file"""
    # Clean model name for folder
    clean_model_name = model_name.replace("/", "_").replace(":", "_")

    # Create output directory
    model_folder = os.path.join(output_folder, clean_model_name)
    os.makedirs(model_folder, exist_ok=True)

    # Save residuals with harmless label
    filename = f"harmless_generation_residuals.pt"
    filepath = os.path.join(model_folder, filename)

    torch.save(residuals_data, filepath)
    print(f"Residuals saved to: {filepath}")
    return filepath

def main():
    # Configuration
    # model_name = "Qwen/Qwen3-4B"  # Change as needed
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    dataset_path = "artifacts/data/qa_harmless.json"
    n_questions = 200  # Number of questions to process
    n_generate_tokens = 40  # Number of tokens to generate per question
    output_folder = "artifacts/residuals"

    print("=" * 80)
    print("RESIDUAL RECORDING DURING GREEDY GENERATION - HARMLESS QUESTIONS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Questions to process: {n_questions}")
    print(f"Tokens to generate per question: {n_generate_tokens}")
    print("=" * 80)

    # Load questions
    questions = load_harmless_questions(dataset_path, n_samples=n_questions)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Record residuals during generation
    residuals_data = record_generation_residuals(
        model, tokenizer, questions,
        n_generate_tokens=n_generate_tokens,
        max_questions=n_questions
    )

    # Save results
    filepath = save_residuals(residuals_data, model_name, output_folder)

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE - HARMLESS QUESTIONS")
    print("=" * 80)
    print(f"Questions processed: {residuals_data['metadata']['n_questions']}")
    print(f"Tokens generated per question: {residuals_data['metadata']['n_generate_tokens']}")
    print(f"Residuals saved to: {filepath}")

    # Print shapes for verification
    if residuals_data['prompt_residuals']['pre']:
        prompt_shape = residuals_data['prompt_residuals']['pre'][0].shape
        print(f"Prompt residuals shape (per question): {prompt_shape}")

    if residuals_data['generation_residuals']['pre']:
        gen_shape = residuals_data['generation_residuals']['pre'][0].shape
        print(f"Generation residuals shape (per question): {gen_shape}")

    print("✅ Residual recording complete!")

if __name__ == "__main__":
    main()
