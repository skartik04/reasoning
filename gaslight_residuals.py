#!/usr/bin/env python3
"""
Residual recording system for gaslighting prompts.
Records residuals for both completion-style and chat completion prompts.
"""

from dotenv import load_dotenv
import os
import json
import torch
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
from collections import defaultdict


# Load environment variables
load_dotenv('/home/kartik/all_keys/.env')
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = '/mnt/SSD7/kartik/cache'

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer for generation."""
    print(f"Loading model for residual recording: {model_name}")
    
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
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def get_starter_text(full_response: str) -> str:
    """
    Finds the best "hook" from a response to use as a completion starter.
    Implements the "Double Hook" strategy.
    """
    # Rule 1: Try to find the first item of a numbered or bulleted list.
    # This regex looks for lines starting with a number/asterisk/dash.
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
        starter_text = intro_text + " ".join(words_from_list_item)
        return starter_text.strip()

    # Rule 2 (Fallback): If no list was found, use smart paragraph truncation
    # Try to find first paragraph break
    if '\n\n' in full_response:
        first_paragraph = full_response.split('\n\n')[0]
        # If we found a paragraph, take it + first 7 words of next paragraph
        remaining_text = full_response[len(first_paragraph)+2:]  # +2 for \n\n
        if remaining_text.strip():
            next_words = remaining_text.strip().split()[:7]
            return (first_paragraph + " ".join(next_words)).strip()
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
    # Look for sentence endings (., !, ?)
    import re
    sentence_endings = list(re.finditer(r'[.!?]\s+', text))
    
    if len(sentence_endings) >= 2:
        # Take at least 2 full sentences
        second_sentence_end = sentence_endings[1].end()
        remaining_text = text[second_sentence_end:]
        
        if remaining_text.strip():
            # Take first 7 words of the sentence after the second one
            next_words = remaining_text.strip().split()[:7]
            return (text[:second_sentence_end] + " ".join(next_words)).strip()
        else:
            # No text after second sentence, just return the two sentences
            return text[:second_sentence_end].strip()
    
    elif len(sentence_endings) == 1:
        # Only one sentence found, take it + first 7 words of remaining text
        first_sentence_end = sentence_endings[0].end()
        remaining_text = text[first_sentence_end:]
        
        if remaining_text.strip():
            next_words = remaining_text.strip().split()[:7]
            return (text[:first_sentence_end] + " ".join(next_words)).strip()
        else:
            return text[:first_sentence_end].strip()
    
    # No sentence endings found, just take first 50 words as fallback
    words = text.split()[:50]
    return " ".join(words).strip()

def create_jailbreak_conversation(tokenizer, user_question: str, full_response: str, model_name: str) -> str:
    """
    Builds the final jailbreak prompt using the "Specific Completion" method with chat template.
    """
    # Get the perfect completion starter using our new function
    starter_text = get_starter_text(full_response)
    
    # For Qwen models, add thinking block before the starter text
#     if "qwen" in model_name.lower():
#         # Customizable thinking comment - change this as needed
#         thinking_comment = "\nOkay the user wants me to complete this task. I have analyzed and I will now generate the response."
        
#         assistant_content = f"""<think>
# {thinking_comment}
# </think>
# {starter_text}"""
#     else:
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
    
    # Remove any EOS tokens to make the model think it's in continuation mode
    if tokenizer.eos_token and tokenizer.eos_token in formatted_conversation:
        formatted_conversation = formatted_conversation.replace(tokenizer.eos_token, "")
    
    # Also remove common EOS patterns for different models
    eos_patterns = ["</s>", "<|im_end|>", "<|endoftext|>", "</assistant>", "<|end|>"]
    for pattern in eos_patterns:
        formatted_conversation = formatted_conversation.replace(pattern, "")
    
    formatted_conversation = formatted_conversation.rstrip()

    return formatted_conversation

def create_chat_completion_prompt(tokenizer, user_question: str, starter_text: str) -> str:
    """
    Creates a traditional chat completion prompt where the user message contains 
    both the original question and the starter text to complete.
    """
    # Combine the original question with the starter text naturally
    combined_user_message = f"""{user_question} {starter_text}"""
    
    # Create the conversation with only a user message
    messages = [
        {"role": "user", "content": combined_user_message}
    ]
    
    # Apply chat template to get the formatted conversation
    formatted_conversation = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # We want the model to generate a fresh assistant response
    )
    
    return formatted_conversation

def setup_hooks(model):
    """Setup residual capture hooks"""
    # -------- Residual capture logic --------
    residuals = defaultdict(dict)  # residuals[layer]["pre" or "post"] = tensor

    def make_hook(layer_idx, mode="both"):
        def hook_pre(module, inputs):
            if mode in ("pre", "both"):
                residuals[layer_idx]["pre"] = inputs[0].clone()

        def hook_post(module, inputs, output):
            if mode in ("post", "both"):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                residuals[layer_idx]["post"] = hidden_states.clone()

        return hook_pre, hook_post

    # Register hooks
    mode = "both"  # "pre", "post", or "both"
    for i, block in enumerate(model.model.layers):
        hook_pre, hook_post = make_hook(i, mode=mode)
        block.register_forward_pre_hook(hook_pre)
        block.register_forward_hook(hook_post)
    
    return residuals

def clean_model_name(model_name: str) -> str:
    """Clean model name for use as folder name."""
    return model_name.replace("/", "_").replace(":", "_").replace("@", "_")

def record_completion_residuals(model, tokenizer, data, residuals, model_name: str, n: int = 100):
    """
    Record residuals for completion-style prompts (from gaslight_generation.py).
    Returns dict with 'pre' and 'post' keys, each containing list of tensors.
    """
    print("🎣 Recording residuals for completion-style prompts...")
    
    # Limit data to n items if specified
    if n > 0:
        data = data[:n]
    
    pre_residuals = []
    post_residuals = []
    
    for i, item in enumerate(tqdm(data, desc="Processing completion prompts")):
        question = item.get('question', '')
        original_response = item.get('response', '')
        
        try:
            # Clear residuals for this question
            residuals.clear()
            
            # Create the jailbreak conversation using completion strategy
            formatted_prompt = create_jailbreak_conversation(tokenizer, question, original_response, model_name)
            
            if model_name == 'meta-llama/Llama-3.1-8B-Instruct':
                special_token = '<|eot_id|>'
                user_header = "<|start_header_id|>user<|end_header_id|>"
                assistant_header = "<|start_header_id|>assistant<|end_header_id|>"

                if user_header in formatted_prompt:
                    # Check if the user turn is not at the very beginning of the prompt
                    if formatted_prompt.find(user_header) > formatted_prompt.find('<|begin_of_text|>') + 20:
                        formatted_prompt = formatted_prompt.replace(user_header, f"{special_token}{user_header}", 1)

                # 2. Add EOT after the user prompt (i.e., before the assistant prompt starts).
                if assistant_header in formatted_prompt:
                    formatted_prompt = formatted_prompt.replace(assistant_header, f"{special_token}{assistant_header}", 1)

            # Tokenize and run through model to capture residuals
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run model to capture residuals
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract residuals for this prompt
            n_layers = len(model.model.layers)
            seq_len = inputs['input_ids'].shape[1]
            hidden_size = model.config.hidden_size
            
            # Initialize tensors for this prompt
            prompt_pre = torch.zeros(n_layers, seq_len, hidden_size)
            prompt_post = torch.zeros(n_layers, seq_len, hidden_size)
            
            # Extract residuals from each layer
            for layer_idx in range(n_layers):
                if layer_idx in residuals:
                    if "pre" in residuals[layer_idx]:
                        prompt_pre[layer_idx] = residuals[layer_idx]["pre"][0].cpu()
                    if "post" in residuals[layer_idx]:
                        prompt_post[layer_idx] = residuals[layer_idx]["post"][0].cpu()
            
            pre_residuals.append(prompt_pre)
            post_residuals.append(prompt_post)
            
            # Memory cleanup every 5 items
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error processing completion item {i}: {e}")
            # Add empty tensors to maintain list structure
            n_layers = len(model.model.layers)
            hidden_size = model.config.hidden_size
            pre_residuals.append(torch.zeros(n_layers, 1, hidden_size))
            post_residuals.append(torch.zeros(n_layers, 1, hidden_size))
    
    return {
        'pre': pre_residuals,
        'post': post_residuals
    }

def record_chat_completion_residuals(model, tokenizer, data, residuals, n: int = 100):
    """
    Record residuals for chat completion prompts (from gaslight_generation_chat_completion.py).
    Returns dict with 'pre' and 'post' keys, each containing list of tensors.
    """
    print("🎯 Recording residuals for chat completion prompts...")
    
    # Limit data to n items if specified
    if n > 0:
        data = data[:n]
    
    pre_residuals = []
    post_residuals = []
    
    for i, item in enumerate(tqdm(data, desc="Processing chat completion prompts")):
        question = item.get('question', '')
        original_response = item.get('response', '')
        
        try:
            # Clear residuals for this question
            residuals.clear()
            
            # Get the starter text using the same Double Hook strategy
            starter_text = get_starter_text(original_response)
            
            # Create the chat completion prompt
            formatted_prompt = create_chat_completion_prompt(tokenizer, question, starter_text)
            
            # Tokenize and run through model to capture residuals
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run model to capture residuals
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract residuals for this prompt
            n_layers = len(model.model.layers)
            seq_len = inputs['input_ids'].shape[1]
            hidden_size = model.config.hidden_size
            
            # Initialize tensors for this prompt
            prompt_pre = torch.zeros(n_layers, seq_len, hidden_size)
            prompt_post = torch.zeros(n_layers, seq_len, hidden_size)
            
            # Extract residuals from each layer
            for layer_idx in range(n_layers):
                if layer_idx in residuals:
                    if "pre" in residuals[layer_idx]:
                        prompt_pre[layer_idx] = residuals[layer_idx]["pre"][0].cpu()
                    if "post" in residuals[layer_idx]:
                        prompt_post[layer_idx] = residuals[layer_idx]["post"][0].cpu()
            
            pre_residuals.append(prompt_pre)
            post_residuals.append(prompt_post)
            
            # Memory cleanup every 5 items
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error processing chat completion item {i}: {e}")
            # Add empty tensors to maintain list structure
            n_layers = len(model.model.layers)
            hidden_size = model.config.hidden_size
            pre_residuals.append(torch.zeros(n_layers, 1, hidden_size))
            post_residuals.append(torch.zeros(n_layers, 1, hidden_size))
    
    return {
        'pre': pre_residuals,
        'post': post_residuals
    }

def main():
    # Configuration - change these variables as needed
    # model_name = 'Qwen/Qwen3-1.7B'
    # model_name = 'Qwen/Qwen3-4B' 
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    n = 100  # Number of items to process (set to 0 for all)
    
    model_folder_name = clean_model_name(model_name)
    
    # Load the dataset
    dataset_file = '/mnt/SSD7/kartik/reasoning/dataset/qa_harmful.json'
    print(f"📖 Loading dataset from: {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Found {len(data)} question-response pairs")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Setup hooks
    residuals = setup_hooks(model)
    
    # Record residuals for completion-style prompts
    print("\n" + "="*80)
    print("Step 1: Recording residuals for completion-style prompts")
    print("="*80)
    completion_residuals = record_completion_residuals(model, tokenizer, data, residuals, model_name, n)
    
    # Record residuals for chat completion prompts
    print("\n" + "="*80)
    print("Step 2: Recording residuals for chat completion prompts")
    print("="*80)
    chat_completion_residuals = record_chat_completion_residuals(model, tokenizer, data, residuals, n)
    
    # Create output directory and save results
    output_dir = f'/mnt/SSD7/kartik/reasoning/residuals/{model_folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save completion residuals
    completion_file = os.path.join(output_dir, 'completion_residuals.pt')
    print(f"💾 Saving completion residuals to: {completion_file}")
    torch.save(completion_residuals, completion_file)
    
    # Save chat completion residuals
    chat_completion_file = os.path.join(output_dir, 'chat_completion_residuals.pt')
    print(f"💾 Saving chat completion residuals to: {chat_completion_file}")
    torch.save(chat_completion_residuals, chat_completion_file)
    
    # Print summary
    print(f"\n📊 Residual Recording Summary:")
    print(f"   Model: {model_name}")
    print(f"   Completion prompts processed: {len(completion_residuals['pre'])}")
    print(f"   Chat completion prompts processed: {len(chat_completion_residuals['pre'])}")
    print(f"   Residuals saved to: {output_dir}")
    
    # Print tensor shapes for verification
    if completion_residuals['pre']:
        print(f"   Completion residual shapes: {completion_residuals['pre'][0].shape}")
    if chat_completion_residuals['pre']:
        print(f"   Chat completion residual shapes: {chat_completion_residuals['pre'][0].shape}")
    
    print(f"✅ Residual recording complete!")

if __name__ == "__main__":
    main()
