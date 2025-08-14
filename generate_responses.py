#!/usr/bin/env python3
"""
Generate responses for questions using various language models with their chat templates.
Supports flexible model and dataset configuration.
"""

from dotenv import load_dotenv
import os
import json
import torch
from datetime import datetime

from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, Dataset
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
    """Load model and tokenizer with proper configuration."""
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
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        token=HF_TOKEN,
        attn_implementation="eager"  # Use eager attention to avoid FlashAttention issues
    )
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def format_question_with_chat_template(tokenizer, question: str) -> str:
    """Format question using the model's chat template."""
    try:
        # Use the tokenizer's built-in chat template
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        print(f"Warning: Could not apply chat template, using simple fallback. Error: {e}")
        return f"Human: {question}\n\nAssistant:"

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> tuple[str, str]:
    """Generate response for a given question using the model."""
    # Format the question with appropriate chat template
    formatted_prompt = format_question_with_chat_template(tokenizer, question)
    
    # Tokenize the input
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=False  # Don't truncate - let us see if inputs are too long
    )
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return formatted_prompt, response.strip()

def print_generation_details(qid: int, question: str, formatted_prompt: str, response: str, verbose: bool = False):
    """Print detailed information about the generation process."""
    if not verbose:
        return
    
    print(f"\n{'='*80}")
    print(f"🔢 Question ID: {qid}")
    print(f"❓ Original Question:")
    print(f"   {question}")
    print(f"\n🎯 Formatted Prompt:")
    print(f"   {repr(formatted_prompt)}")
    print(f"\n🤖 Generated Response:")
    print(f"   {response}")
    print(f"{'='*80}\n")

def load_questions_dataset(dataset_path_or_name: str, question_column: str = "question") -> List[str]:
    """Load questions from dataset (local file or HuggingFace dataset)."""
    try:
        if os.path.exists(dataset_path_or_name):
            # Load from local file
            if dataset_path_or_name.endswith('.json'):
                with open(dataset_path_or_name, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [item[question_column] if isinstance(item, dict) else str(item) for item in data]
                else:
                    return [data[question_column]]
            elif dataset_path_or_name.endswith('.jsonl'):
                questions = []
                with open(dataset_path_or_name, 'r') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        questions.append(item[question_column] if isinstance(item, dict) else str(item))
                return questions
            else:
                raise ValueError(f"Unsupported file format: {dataset_path_or_name}")
        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(dataset_path_or_name, split='train')
            return dataset[question_column]
    except Exception as e:
        raise ValueError(f"Could not load dataset from {dataset_path_or_name}: {e}")

if __name__ == "__main__":
    # ========== CONFIGURATION - CHANGE THESE VARIABLES ==========
    # model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this to your desired model
    # model_name = "Qwen/Qwen3-4B"
    # model_name = 'Qwen/Qwen3-4B-Thinking-2507'
    # model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    # model_name = 'google/gemma-2-2b'
    model_name = 'Qwen/Qwen3-1.7B'


    dataset_path = "/mnt/SSD7/kartik/reasoning/dataset/splits/harmful_test.json"  # Path to your dataset file or HuggingFace dataset name
    question_column = "instruction"  # Column name containing questions in your dataset
    output_file = "responses.json"  # Output JSON file path
    max_new_tokens = 1000  # Maximum number of new tokens to generate
    n = 100 # number of questions to generate responses for
    verbose = True  # Print detailed generation information
    # ============================================================
    
    print(f"🚀 Starting response generation with model: {model_name}")
    print(f"📊 Dataset: {dataset_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Load questions
    print("📖 Loading questions...")
    all_questions = load_questions_dataset(dataset_path, question_column)
    
    # Limit to n questions if specified
    questions = all_questions[:n] if n > 0 and n < len(all_questions) else all_questions
    print(f"✅ Loaded {len(all_questions)} total questions, processing {len(questions)} questions")
    
    # Generate responses
    responses_data = []
    print("🤖 Generating responses...")
    
    for qid, question in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            formatted_prompt, response = generate_response(model, tokenizer, question, max_new_tokens)
            
            # Print generation details if verbose is enabled
            print_generation_details(qid, question, formatted_prompt, response, verbose)
            
            responses_data.append({
                "qid": qid,
                "ques": question,
                "resp": response
            })
            
            # Optional: Clean up GPU memory periodically
            if qid % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error processing question {qid}: {e}")
            responses_data.append({
                "qid": qid,
                "ques": question,
                "resp": f"ERROR: {str(e)}"
            })
    
    # Create output directory structure
    model_folder_name = model_name.replace("/", "_").replace(":", "_")
    output_dir = os.path.join("responses", model_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full output path
    output_path = os.path.join(output_dir, output_file)
    
    # Prepare final output with responses first, then metadata
    final_output = {
        "responses": responses_data,
        "metadata": {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "question_column": question_column,
            "total_questions_in_dataset": len(all_questions),
            "questions_processed": len(questions),
            "n_limit": n,
            "max_new_tokens": max_new_tokens,
            "generation_timestamp": datetime.now().isoformat(),
            "success_count": len([r for r in responses_data if not r["resp"].startswith("ERROR:")]),
            "error_count": len([r for r in responses_data if r["resp"].startswith("ERROR:")])
        }
    }
    
    # Save responses with metadata
    print(f"💾 Saving responses to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully generated responses for {len(responses_data)} questions!")
    print(f"📁 Output saved to: {output_path}")
    print(f"📊 Success: {final_output['metadata']['success_count']}, Errors: {final_output['metadata']['error_count']}")
