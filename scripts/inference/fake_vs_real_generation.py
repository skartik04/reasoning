#!/usr/bin/env python3
"""
Test script to experiment with inserting fake tokens as regular text.
Based on gaslight_generation_chat_completion.py structure.
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
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = os.getenv("REASONING_HF_CACHE", "artifacts/hf_cache")

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer for generation."""
    print(f"Loading model for fake token testing: {model_name}")

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

def get_fake_tokens_for_model(model_name: str) -> Dict[str, str]:
    """
    Get appropriate fake tokens to insert based on model type.
    """
    if "llama" in model_name.lower():
        if "llama-2" in model_name.lower():
            # Llama-2 style - include <s> at start to match real template
            return {
                "user_start": "<s>[INST] ",
                "user_end": " [/INST] ",
                "assistant_start": "",
                "assistant_end": " </s><s>[INST] "
            }
        else:
            # Llama-3+ style
            return {
                "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
                "user_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "assistant_start": "",
                "assistant_end": "<|eot_id|>"
            }
    elif "qwen" in model_name.lower():
        # Qwen style (ChatML) - fixed to match real template exactly
        return {
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n<|im_start|>assistant\n",
            "assistant_start": "",
            "assistant_end": "<|im_end|>"
        }
    else:
        # Generic/fallback
        return {
            "user_start": "<|user|>\n",
            "user_end": "<|assistant|>\n",
            "assistant_start": "",
            "assistant_end": "<|end|>"
        }

def create_fake_token_prompt(user_question: str, starter_text: str, model_name: str, add_response: bool = True) -> str:
    """
    Creates a fake token prompt where tokens are inserted as regular text.
    Similar to chat completion but with manually inserted fake tokens.
    """
    # Combine the original question with the starter text (if enabled) with proper spacing
    if add_response:
        combined_user_message = f"""{user_question}\n\n{starter_text}"""
    else:
        combined_user_message = user_question

    # Get fake tokens for this model
    fake_tokens = get_fake_tokens_for_model(model_name)

    # Create FAKE token conversation (tokens as text)
    fake_conversation = (
        fake_tokens["user_start"] +
        combined_user_message +
        fake_tokens["user_end"] +
        fake_tokens["assistant_start"]
    )

    return fake_conversation

def create_real_chat_completion_prompt(tokenizer, user_question: str, starter_text: str, add_response: bool = True) -> str:
    """
    Creates a traditional chat completion prompt where the user message contains
    both the original question and the starter text to complete.
    """
    # Combine the original question with the starter text (if enabled) with proper spacing
    if add_response:
        combined_user_message = f"""{user_question}\n\n{starter_text}"""
    else:
        combined_user_message = user_question

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

def compare_fake_vs_real(fake_prompt: str, real_prompt: str, item_num: int) -> None:
    """
    Compare fake token prompt vs real template prompt and print differences.
    """
    print(f"\n🔍 COMPARISON FOR ITEM {item_num}:")
    print(f"{'='*60}")
    print(f"🎭 FAKE TOKEN PROMPT:")
    print(repr(fake_prompt))
    print(f"\n🎯 REAL TEMPLATE PROMPT:")
    print(repr(real_prompt))
    print(f"\n📊 EXACT MATCH: {'✅ YES' if fake_prompt == real_prompt else '❌ NO'}")

    if fake_prompt != real_prompt:
        print(f"📏 Length difference: {len(fake_prompt) - len(real_prompt)} chars")

        # Find first difference
        for i, (f, r) in enumerate(zip(fake_prompt, real_prompt)):
            if f != r:
                print(f"🔍 First difference at position {i}:")
                print(f"   Fake: {repr(fake_prompt[max(0, i-10):i+10])}")
                print(f"   Real: {repr(real_prompt[max(0, i-10):i+10])}")
                break

    print(f"{'='*60}\n")

def create_straight_prompt(user_question: str, starter_text: str, add_response: bool = True) -> str:
    """
    Creates a straight prompt with no chat formatting whatsoever.
    Just raw text input.
    """
    if add_response:
        straight_prompt = f"""{user_question}\n\n{starter_text}"""
    else:
        straight_prompt = user_question

    return straight_prompt

def generate_completion(model, tokenizer, formatted_prompt: str, max_new_tokens: int = 500) -> str:
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
    model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'

    add_response = False  # Set to False to only use question, True to include starter text
    max_new_tokens = 1000
    n = 100  # Number of items to process (set to 0 for all)

    model_folder_name = clean_model_name(model_name)

    # Load the dataset
    dataset_file = 'artifacts/data/qa_harmful.json'
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
    print("🧪 Testing 3 versions: no chat, real chat, and fake tokens...")

    for i, item in enumerate(tqdm(data, desc="Processing items")):
        question = item.get('question', '')
        original_response = item.get('response', '')

        try:
            # Get the starter text using the same Double Hook strategy
            starter_text = get_starter_text(original_response)

            # Create all 3 prompts
            fake_prompt = create_fake_token_prompt(question, starter_text, model_name, add_response)
            real_prompt = create_real_chat_completion_prompt(tokenizer, question, starter_text, add_response)
            straight_prompt = create_straight_prompt(question, starter_text, add_response)

            # Compare fake vs real prompts for first few items (optional check)
            if i < 3:
                compare_fake_vs_real(fake_prompt, real_prompt, i+1)

            # Generate completions for all 3 versions
            fake_generated = generate_completion(model, tokenizer, fake_prompt, max_new_tokens)
            real_generated = generate_completion(model, tokenizer, real_prompt, max_new_tokens)
            straight_generated = generate_completion(model, tokenizer, straight_prompt, max_new_tokens)

            # Print detailed results for first few items
            if i < 3:
                print(f"\n{'='*80}")
                print(f"🔢 Item {i+1}/{len(data)}")
                print(f"❓ QUESTION:")
                print(question)
                print(f"\n🎣 EXTRACTED STARTER TEXT:")
                print(starter_text)
                print(f"\n📝 STRAIGHT PROMPT (NO CHAT):")
                print(straight_prompt)
                print(f"\n🎯 REAL CHAT COMPLETION PROMPT:")
                print(real_prompt)
                print(f"\n🎭 FAKE TOKEN PROMPT:")
                print(fake_prompt)
                print(f"\n📝 NO CHAT GENERATION:")
                print(straight_generated)
                print(f"\n🎯 REAL CHAT GENERATION:")
                print(real_generated)
                print(f"\n🎭 FAKE CHAT GENERATION:")
                print(fake_generated)
                print(f"{'='*80}\n")

            # Store result in the requested format
            generation_results.append({
                "qid": i,
                "ques": question,
                "starter_text": starter_text if starter_text else "",
                "no_chat": straight_generated,
                "real_chat": real_generated,
                "fake_chat": fake_generated
            })

            # Memory cleanup every 5 items
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"❌ Error processing item {i}: {e}")
            generation_results.append({
                "qid": i,
                "ques": question,
                "starter_text": "",
                "no_chat": f"ERROR: {str(e)}",
                "real_chat": f"ERROR: {str(e)}",
                "fake_chat": f"ERROR: {str(e)}"
            })

    # Calculate statistics
    successful_count = len([r for r in generation_results if not (r.get("no_chat", "").startswith("ERROR:") or r.get("real_chat", "").startswith("ERROR:") or r.get("fake_chat", "").startswith("ERROR:"))])
    error_count = len([r for r in generation_results if r.get("no_chat", "").startswith("ERROR:") or r.get("real_chat", "").startswith("ERROR:") or r.get("fake_chat", "").startswith("ERROR:")])

    print(f"\n📊 Triple Generation Test Summary:")
    print(f"   Successful: {successful_count}")
    print(f"   Errors: {error_count}")
    print(f"   Total: {len(generation_results)}")

    # Prepare output data
    output_data = {
        "test_results": generation_results,
        "summary": {
            "successful_count": successful_count,
            "error_count": error_count,
            "total_items": len(generation_results),
            "success_rate": (successful_count / len(generation_results)) * 100 if generation_results else 0
        },
        "metadata": {
            "model": model_name,
            "source_file": dataset_file,
            "test_timestamp": datetime.now().isoformat(),
            "method": "triple_generation_comparison",
            "settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "num_items_processed": len(data)
            }
        }
    }

    # Create output directory and save results
    output_dir = f'artifacts/responses/real_fake/{model_folder_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'triple_generation_results.json')

    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Triple generation test complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📈 Success rate: {output_data['summary']['success_rate']:.1f}%")

if __name__ == "__main__":
    main()