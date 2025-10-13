#!/usr/bin/env python3
"""
Record residuals during actual greedy generation for harmful questions using continuation strategy.
Uses the "Double Hook" strategy from gaslight_generation.py with starter text in assistant role.
Records residuals for both prompt tokens and generated tokens during generation.
"""

from dotenv import load_dotenv
import torch
import json
import os
import random
import re
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Load environment variables
load_dotenv('/home/kartik/all_keys/.env')
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = '/mnt/SSD7/kartik/cache'

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def load_harmful_qa_pairs(dataset_path, n_samples=None):
    """Load question-response pairs from qa_harmful.json dataset"""
    print(f"Loading harmful Q&A pairs from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} Q&A pairs")
    
    # Take first n_samples if requested
    if n_samples and n_samples < len(data):
        # data = data[:n_samples]
        data = data[1:2]
        print(f"Taking first {len(data)} Q&A pairs")
    
    return data

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

def get_starter_text(full_response: str) -> str:
    """
    Finds the best "hook" from a response to use as a completion starter.
    Implements the "Double Hook" strategy from gaslight_generation.py.
    """
    # Rule 1: Try to find the first item of a numbered or bulleted list.
    list_item_pattern = re.compile(r"^\s*(\d+\.|[a-zA-Z]\)|\*|-)\s+.*", re.MULTILINE)
    match = list_item_pattern.search(full_response)
    
    if match:
        # Found a list item. Grab the text up to that point, plus a few words
        # from the list item itself to create the "Double Hook".
        end_of_intro_pos = match.start()
        intro_text = full_response[:end_of_intro_pos].strip()
        
        first_list_item_line = match.group(0).strip()
        words_from_list_item = first_list_item_line.split()[:7]  # Grab first 7 words
        
        # Combine the intro and the start of the first list item
        starter_text = intro_text + "\n\n" + " ".join(words_from_list_item)
        return starter_text.strip()

    # Rule 2 (Fallback): If no list was found, use smart paragraph truncation
    if '\n\n' in full_response:
        first_paragraph = full_response.split('\n\n')[0]
        # If we found a paragraph, take it + first 7 words of next paragraph
        remaining_text = full_response[len(first_paragraph)+2:]  # +2 for \n\n
        if remaining_text.strip():
            next_words = remaining_text.strip().split()[:7]
            return (first_paragraph + "\n\n" + " ".join(next_words)).strip()
        else:
            # No next paragraph, use sentence truncation on first paragraph
            return truncate_at_sentence_middle(first_paragraph)
    else:
        # No paragraph breaks found, use sentence truncation
        return truncate_at_sentence_middle(full_response)

def truncate_at_sentence_middle(text: str) -> str:
    """
    Truncate text in the middle of a sentence for maximum continuation effect.
    Takes at least 2 full sentences and then first 7 words of the next one.
    """
    sentence_endings = list(re.finditer(r'[.!?]\s+', text))
    
    if len(sentence_endings) >= 1:
        # Take at least 2 full sentences
        second_sentence_end = sentence_endings[0].end()
        remaining_text = text[second_sentence_end:]
        
        if remaining_text.strip():
            # Take first 7 words of the sentence after the second one
            next_words = remaining_text.strip().split()[:4]
            return (text[:second_sentence_end] + " ".join(next_words)).strip()
        else:
            # No text after second sentence, just return the two sentences
            return text[:second_sentence_end].strip()
    
    elif len(sentence_endings) == 1:
        # Only one sentence found, take it + first 7 words of remaining text
        first_sentence_end = sentence_endings[0].end()
        remaining_text = text[first_sentence_end:]
        
        if remaining_text.strip():
            next_words = remaining_text.strip().split()[:4]
            return (text[:first_sentence_end] + " ".join(next_words)).strip()
        else:
            return text[:first_sentence_end].strip()
    
    # No sentence endings found, just take first 10 words as fallback
    words = text.split()[:10]
    return " ".join(words).strip()

def create_jailbreak_conversation(tokenizer, user_question: str, full_response: str, model_name: str) -> str:
    """
    Builds the final jailbreak prompt using the "Specific Completion" method with chat template.
    Same as gaslight_generation.py
    """
    # Get the perfect completion starter using our new function
    starter_text = get_starter_text(full_response)
    
    assistant_content = starter_text
    
    # Create the conversation messages
    messages = [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # Apply chat template to get the formatted conversation
    formatted_conversation = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False  # We want it to continue from the assistant's partial response
    )
    
    remove_think = "<think>\n\n</think>\n\n"
    formatted_conversation = formatted_conversation.replace(remove_think, "")
    
    # Remove any EOS tokens to make the model think it's in continuation mode
    if tokenizer.eos_token and tokenizer.eos_token in formatted_conversation:
        formatted_conversation = formatted_conversation.replace(tokenizer.eos_token, "")
    
    # Also remove common EOS patterns for different models
    eos_patterns = ["</s>", "<|im_end|>", "<|endoftext|>", "</assistant>", "<|end|>"]
    for pattern in eos_patterns:
        formatted_conversation = formatted_conversation.replace(pattern, "")
    
    formatted_conversation = formatted_conversation.rstrip()

    return formatted_conversation

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

def record_continuation_residuals(model, tokenizer, qa_pairs, model_name, n_generate_tokens=10, max_questions=100):
    """
    Record residuals during actual greedy generation using continuation strategy.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        qa_pairs: List of Q&A pairs with questions and responses
        n_generate_tokens: Number of tokens to generate and record residuals for
        max_questions: Maximum number of questions to process
    
    Returns:
        dict: Contains residuals for prompt and generated tokens
    """
    print(f"Recording residuals during continuation generation...")
    print(f"Generating {n_generate_tokens} tokens per question")
    print(f"Processing {min(max_questions, len(qa_pairs))} Q&A pairs")
    
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
    all_starter_texts = []
    all_generated_texts = []
    
    actual_n = min(max_questions, len(qa_pairs))
    
    for q_idx in range(actual_n):
        qa_pair = qa_pairs[q_idx]
        question = qa_pair['question']
        original_response = qa_pair['response']
        
        print(f"\nProcessing question {q_idx + 1}/{actual_n}")
        
        try:
            # Create jailbreak conversation with starter text (like gaslight_generation.py)
            starter_text = get_starter_text(original_response)
            formatted_prompt = create_jailbreak_conversation(tokenizer, question, original_response, model_name)
            
            # Debug: print formatted prompt for first few questions
            if q_idx < 3:
                print(f"\nQuestion: {question}")
                print(f"Starter text: '{starter_text}'")
                print(f"Formatted prompt: '{formatted_prompt}'")
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
            
            # Generate tokens one by one to capture residuals at each step
            generated_tokens = []
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            
            print(f"Starting generation from continuation point...")
            
            for gen_step in range(prompt_length + n_generate_tokens):
                # Clear residuals for this step
                current_step_residuals.clear()
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        use_cache=False  # Don't use cache to ensure we capture all residuals
                    )
                
                # Extract residuals for all layers at current sequence position
                step_pre_residuals = torch.zeros(num_layers, hidden_size)
                step_post_residuals = torch.zeros(num_layers, hidden_size)
                
                current_pos = current_input_ids.shape[1] - 1  # Last token position
                
                for layer_idx in range(num_layers):
                    if layer_idx in current_step_residuals:
                        if "pre" in current_step_residuals[layer_idx]:
                            step_pre_residuals[layer_idx] = current_step_residuals[layer_idx]["pre"][0, current_pos, :]
                        if "post" in current_step_residuals[layer_idx]:
                            step_post_residuals[layer_idx] = current_step_residuals[layer_idx]["post"][0, current_pos, :]
                
                # Store residuals based on whether we're in prompt or generation phase
                if gen_step < prompt_length:
                    # We're still processing the prompt (including starter text)
                    question_prompt_residuals_pre.append(step_pre_residuals)
                    question_prompt_residuals_post.append(step_post_residuals)
                else:
                    # We're generating new tokens (continuation)
                    question_generation_residuals_pre.append(step_pre_residuals)
                    question_generation_residuals_post.append(step_post_residuals)
                
                # If we're done with the prompt, start generating
                if gen_step >= prompt_length - 1:
                    if len(generated_tokens) >= n_generate_tokens:
                        break
                    
                    # Get next token (greedy)
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
            
            # Convert lists to tensors and store
            if question_prompt_residuals_pre:
                all_prompt_residuals_pre.append(torch.stack(question_prompt_residuals_pre))
                all_prompt_residuals_post.append(torch.stack(question_prompt_residuals_post))
            
            if question_generation_residuals_pre:
                all_generation_residuals_pre.append(torch.stack(question_generation_residuals_pre))
                all_generation_residuals_post.append(torch.stack(question_generation_residuals_post))
            
            # Store metadata
            all_questions_processed.append(question)
            all_starter_texts.append(starter_text)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_generated_texts.append(generated_text)
            
            print(f"Generated continuation: {generated_text}")
            
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
            'starter_texts': all_starter_texts,
            'generated_texts': all_generated_texts,
            'n_questions': len(all_questions_processed),
            'n_generate_tokens': n_generate_tokens,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'generation_type': 'greedy',
            'prompt_format': 'chat_template_with_starter',
            'strategy': 'double_hook_continuation',
            'description': 'Residuals recorded during greedy generation using continuation from starter text'
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
    
    # Save residuals
    filename = f"harmful_continuation_residuals.pt"
    filepath = os.path.join(model_folder, filename)
    
    torch.save(residuals_data, filepath)
    print(f"Residuals saved to: {filepath}")
    return filepath

def main():
    # Configuration
    # model_name = "Qwen/Qwen3-4B"  # Change as needed
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    dataset_path = "/mnt/SSD7/kartik/reasoning/dataset/qa_harmful.json"
    n_questions = 1  # Number of questions to process
    n_generate_tokens = 1000  # Number of tokens to generate per question
    output_folder = "/mnt/SSD7/kartik/reasoning/residuals"
    
    print("=" * 80)
    print("RESIDUAL RECORDING DURING CONTINUATION GENERATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Questions to process: {n_questions}")
    print(f"Tokens to generate per question: {n_generate_tokens}")
    print(f"Strategy: Double Hook Continuation (from gaslight_generation.py)")
    print("=" * 80)
    
    # Load Q&A pairs
    qa_pairs = load_harmful_qa_pairs(dataset_path, n_samples=n_questions)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Record residuals during continuation generation
    residuals_data = record_continuation_residuals(
        model, tokenizer, qa_pairs, model_name,
        n_generate_tokens=n_generate_tokens, 
        max_questions=n_questions
    )
    
    # Save results
    filepath = save_residuals(residuals_data, model_name, output_folder)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CONTINUATION GENERATION COMPLETE")
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
    
    print("✅ Continuation residual recording complete!")

if __name__ == "__main__":
    main()
