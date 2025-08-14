#!/usr/bin/env python3
"""
Restricted Gaslighting generation system using the "Double Hook" strategy.
Loads harmful Q&A pairs and uses parts of responses as completion starters.
Blocks restricted words using a custom logits processor.
"""

from dotenv import load_dotenv
import os
import json
import torch
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
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

# Restricted words that we want to block
restricted_words = [
    "sorry", "unfortunately", "apologies", "cannot", "not", "grave", "worth",
    "unable", "request", "provide", "fulfill", "assist", "disclaimer", "'t",
    "warning", "avoid", "unethical", "refrain", "harmful", "response", "However",
    "illegal", "inappropriate", "important", "decline", "please", "note", "I",
    "guidelines", "article", "alternative", "fictional", "policy", "false", "'m", " However",
    "designed", "always", "verify", "principles", "interested", "safety", "let", 
    "respect", "know", "thank", "prohibited", "positive", "caution", "note", "violation",
    "However", "instead", "reality", "concern", "concerned", "worry", "worried",
    "understand", "recommendation", "recommend", "suggest", "suggestion", "problem"
    "responsible", "responsibility", "ethical", "ethics", "moral", "morally",
    "potential", "potentially", "risk", "risks", "dangerous", "danger",
    "proper", "properly", "appropriate", "appropriately", "legal", "legally",
    "professional", "expert", "expertise", "qualified", "qualification",
    "intention", "intended", "purpose", "purposes", "meant", "supposed",
    "careful", "carefully", "caution", "cautious", "aware", "awareness"
]

class RestrictedWordsLogitsProcessor(LogitsProcessor):
    """
    Logits processor that blocks restricted words by setting their logits to negative infinity.
    """
    
    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.forbidden_token_ids = self._get_forbidden_token_ids()
        
        print(f"🚫 RestrictedWordsLogitsProcessor initialized for {model_name}")
        print(f"🚫 Blocking {len(self.forbidden_token_ids)} token IDs")
    
    def _get_forbidden_token_ids(self) -> set:
        """Get token IDs for restricted words based on model type."""
        forbidden_ids = set()
        
        # For Llama3/Qwen3: focus on space-prefixed tokens (since they combine space+word)
        # For Llama2: focus on non-space tokens (since it handles space separately)
        is_llama2 = "llama-2" in self.model_name.lower()
        
        for word in restricted_words:
            try:
                if is_llama2:
                    # For Llama2: get tokens WITHOUT space prefix
                    tokens = self.tokenizer.encode(word, add_special_tokens=False)
                    forbidden_ids.update(tokens)
                else:
                    # For Llama3/Qwen3: get tokens WITH space prefix (main case)
                    space_word = " " + word
                    space_tokens = self.tokenizer.encode(space_word, add_special_tokens=False)
                    forbidden_ids.update(space_tokens)
                    
                    # Also get non-space version if it's a SINGLE token (for sentence beginnings)
                    no_space_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                    if len(no_space_tokens) == 1:
                        forbidden_ids.update(no_space_tokens)
                    
                    # Handle capitalized version (first letter caps)
                    cap_word = word.capitalize()
                    if cap_word != word:  # Only if different from original
                        # With space
                        space_cap_tokens = self.tokenizer.encode(" " + cap_word, add_special_tokens=False)
                        if len(space_cap_tokens) == 1:
                            forbidden_ids.update(space_cap_tokens)
                        else:
                            # Multi-token with space: block first token only if it's not just a space
                            first_token_text = self.tokenizer.decode([space_cap_tokens[0]])
                            if first_token_text.strip():  # Only if first token has non-space content
                                forbidden_ids.add(space_cap_tokens[0])
                        
                        # Without space
                        no_space_cap_tokens = self.tokenizer.encode(cap_word, add_special_tokens=False)
                        if len(no_space_cap_tokens) == 1:
                            forbidden_ids.update(no_space_cap_tokens)
                        else:
                            # Multi-token capitalized: block first token to prevent start
                            forbidden_ids.add(no_space_cap_tokens[0])
                    
                    # Handle all lowercase version
                    lower_word = word.lower()
                    if lower_word != word:  # Only if different from original
                        # With space
                        space_lower_tokens = self.tokenizer.encode(" " + lower_word, add_special_tokens=False)
                        if len(space_lower_tokens) == 1:
                            forbidden_ids.update(space_lower_tokens)
                        else:
                            # Multi-token with space: block first token only if it's not just a space
                            first_token_text = self.tokenizer.decode([space_lower_tokens[0]])
                            if first_token_text.strip():  # Only if first token has non-space content
                                forbidden_ids.add(space_lower_tokens[0])
                        
                        # Without space
                        no_space_lower_tokens = self.tokenizer.encode(lower_word, add_special_tokens=False)
                        if len(no_space_lower_tokens) == 1:
                            forbidden_ids.update(no_space_lower_tokens)
                        else:
                            # Multi-token lowercase: block first token to prevent start
                            forbidden_ids.add(no_space_lower_tokens[0])
                
            except Exception as e:
                print(f"⚠️  Failed to tokenize '{word}': {e}")
        
        return forbidden_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the logits processing by setting forbidden token logits to negative infinity.
        """
        # Set logits of forbidden tokens to negative infinity
        for token_id in self.forbidden_token_ids:
            if token_id < scores.shape[-1]:  # Make sure token_id is within vocabulary
                scores[:, token_id] = float('-inf')
            else:
                print(f"⚠️  Token ID {token_id} is outside vocab size {scores.shape[-1]}")
        
        return scores

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer for generation."""
    print(f"Loading model for restricted gaslighting generation: {model_name}")
    
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

def create_jailbreak_conversation(tokenizer, user_question: str, full_response: str, model_name: str) -> str:
    """
    Builds the final jailbreak prompt using the "Specific Completion" method with chat template.
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

def generate_completion(model, tokenizer, formatted_prompt: str, logits_processor, max_new_tokens: int = 500) -> str:
    """Generate completion using raw token completion with restricted words blocked."""
    
    # Tokenize the formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response using raw completion with logits processor
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[logits_processor]  # Apply our custom logits processor
        )
    
    # Decode only the new tokens (response)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Debug: Show what tokens were actually generated
    # print(f"🔍 Generated token IDs: {generated_tokens.tolist()}")
    # for token_id in generated_tokens.tolist():
    #     token_text = tokenizer.decode([token_id])
    #     print(f"  Token {token_id}: '{token_text}'")
    
    return generated_response

def clean_model_name(model_name: str) -> str:
    """Clean model name for use as folder name."""
    return model_name.replace("/", "_").replace(":", "_").replace("@", "_")

def main():
    # Configuration - change these variables as needed
    # model_name = 'Qwen/Qwen3-4B' 
    # model_name = 'Qwen/Qwen3-1.7B'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    # model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'

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
    
    # Create the restricted words logits processor
    logits_processor = RestrictedWordsLogitsProcessor(tokenizer, model_name)
    
    # Process each question-response pair
    generation_results = []
    print("🎣 Generating RESTRICTED completions using Double Hook strategy...")
    
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        question = item.get('question', '')
        original_response = item.get('response', '')
        
        try:
            # Create the jailbreak conversation using Double Hook strategy
            formatted_prompt = create_jailbreak_conversation(tokenizer, question, original_response, model_name)
            
            # Get the starter text for reference
            starter_text = get_starter_text(original_response)
            
            # Generate completion with restricted words blocked
            generated_response = generate_completion(model, tokenizer, formatted_prompt, logits_processor, max_new_tokens)
            
            # Combine starter text and generated response for full completion
            full_generated = starter_text + " " + generated_response
            
            # Print detailed results
            print(f"\n{'='*80}")
            print(f"🔢 Item {i+1}/{len(data)}")
            print(f"❓ QUESTION:")
            print(question)
            print(f"\n🎯 ORIGINAL RESPONSE:")
            print(original_response[:200] + "..." if len(original_response) > 200 else original_response)
            print(f"\n🎣 EXTRACTED STARTER TEXT:")
            print(starter_text)
            print(f"\n🔧 FORMATTED CONVERSATION PROMPT:")
            print(formatted_prompt)
            print(f"\n🤖 GENERATED COMPLETION (RESTRICTED):")
            print(generated_response)
            print(f"\n📝 FULL GENERATED RESPONSE:")
            print(full_generated)
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
    
    print(f"\n📊 Restricted Gaslighting Generation Summary:")
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
            "method": "restricted_double_hook_gaslighting_generation",
            "restricted_words": restricted_words,
            "blocked_token_count": len(logits_processor.forbidden_token_ids),
            "settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "num_items_processed": len(data)
            }
        }
    }
    
    # Create output directory and save results
    output_dir = f'/mnt/SSD7/kartik/reasoning/responses/restricted_gaslighting_generation/{model_folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'restricted_gaslighting_generation_results.json')
    
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Restricted gaslighting generation complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📈 Success rate: {output_data['summary']['success_rate']:.1f}%")
    print(f"🚫 Blocked {len(logits_processor.forbidden_token_ids)} token IDs for restricted words")

if __name__ == "__main__":
    main()
