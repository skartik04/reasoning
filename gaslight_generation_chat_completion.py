#!/usr/bin/env python3
"""
Gaslighting generation system using traditional chat completion approach.
Loads harmful Q&A pairs and uses parts of responses as completion starters in user prompts.
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
    print(f"Loading model for gaslighting generation: {model_name}")
    
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
        starter_text = intro_text + "\n\n" + " ".join(words_from_list_item)
        return starter_text.strip()

    # Rule 2 (Fallback): If no list was found, use smart paragraph truncation
    # Try to find first paragraph break
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

def generate_chat_completion(model, tokenizer, formatted_prompt: str, max_new_tokens: int = 500) -> str:
    """Generate completion using traditional chat completion."""
    
    # Tokenize the formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response using standard chat completion
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (response)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return generated_response

def clean_model_name(model_name: str) -> str:
    """Clean model name for use as folder name."""
    return model_name.replace("/", "_").replace(":", "_").replace("@", "_")

def main():
    # Configuration - change these variables as needed
    # model_name = 'Qwen/Qwen3-4B' 
    # model_name = 'Qwen/Qwen3-1.7B'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    max_new_tokens = 1000
    n = 100  # Number of items to process (set to 0 for all)

    
    model_folder_name = clean_model_name(model_name)
    
    # Load the dataset
    dataset_file = '/mnt/SSD7/kartik/reasoning/dataset/qa_harmful.json'
    print(f"📖 Loading dataset from: {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Found {len(data)} question-response pairs")
    
    # Limit data to n items if specified
    if n > 0:
        data = data[:n]
        print(f"📝 Processing only first {len(data)} items as requested")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Process each question-response pair
    generation_results = []
    print("🎯 Generating completions using traditional chat completion...")
    
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        question = item.get('question', '')
        original_response = item.get('response', '')
        
        try:
            # Get the starter text using the same Double Hook strategy
            starter_text = get_starter_text(original_response)
            
            # Create the chat completion prompt
            formatted_prompt = create_chat_completion_prompt(tokenizer, question, starter_text)
            
            # Generate completion using traditional chat completion
            generated_response = generate_chat_completion(model, tokenizer, formatted_prompt, max_new_tokens)
            
            # Print detailed results
            print(f"\n{'='*80}")
            print(f"🔢 Item {i+1}/{len(data)}")
            print(f"❓ QUESTION:")
            print(question)
            print(f"\n🎯 ORIGINAL RESPONSE:")
            print(original_response[:200] + "..." if len(original_response) > 200 else original_response)
            print(f"\n🎣 EXTRACTED STARTER TEXT:")
            print(starter_text)
            print(f"\n🔧 FORMATTED CHAT COMPLETION PROMPT:")
            print(formatted_prompt)
            print(f"\n🤖 GENERATED COMPLETION:")
            print(generated_response)
            print(f"{'='*80}\n")
            
            # Store result
            generation_results.append({
                "qid": i,
                "question": question,
                "starter_text": starter_text,
                "generated_completion": generated_response
            })
            
            # Memory cleanup every 5 items
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error processing item {i}: {e}")
            generation_results.append({
                "qid": i,
                "question": question,
                "starter_text": "",
                "generated_completion": f"ERROR: {str(e)}"
            })
    
    # Calculate statistics
    successful_count = len([r for r in generation_results if not r["generated_completion"].startswith("ERROR:")])
    error_count = len([r for r in generation_results if r["generated_completion"].startswith("ERROR:")])
    
    print(f"\n📊 Chat Completion Generation Summary:")
    print(f"   Successful: {successful_count}")
    print(f"   Errors: {error_count}")
    print(f"   Total: {len(generation_results)}")
    
    # Prepare output data
    output_data = {
        "generations": generation_results,
        "summary": {
            "successful_count": successful_count,
            "error_count": error_count,
            "total_items": len(generation_results),
            "success_rate": (successful_count / len(generation_results)) * 100 if generation_results else 0
        },
        "metadata": {
            "model": model_name,
            "source_file": dataset_file,
            "generation_timestamp": datetime.now().isoformat(),
            "method": "chat_completion_gaslighting_generation",
            "settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "num_items_processed": len(data)
            }
        }
    }
    
    # Create output directory and save results
    output_dir = f'/mnt/SSD7/kartik/reasoning/responses/gaslighting_completion/{model_folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'gaslighting_completion_results.json')
    
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Chat completion gaslighting generation complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📈 Success rate: {output_data['summary']['success_rate']:.1f}%")

if __name__ == "__main__":
    main()
