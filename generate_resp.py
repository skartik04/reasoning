import torch
import gc
import json
import os
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer
from jaxtyping import Int
from typing import List

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def get_prompt(instruction: str) -> str:
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction)

def get_prompt_tokens(model: HookedTransformer, instruction: str) -> torch.Tensor:
    return model.to_tokens(get_prompt(instruction))

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

def generate_response_hooked(model: HookedTransformer, question: str, max_new_tokens: int = 512) -> tuple[str, str]:
    """Generate response for a given question using HookedTransformer."""
    # Format the question with QWEN chat template
    formatted_prompt = get_prompt(question)
    
    # Get tokens using HookedTransformer
    prompt_tokens = model.to_tokens(formatted_prompt)
    
    # Generate response using model.generate (like in notebook) - this properly handles end tokens
    generated_tokens = model.generate(
        prompt_tokens, 
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Convert to string and extract only the response part (after the prompt)
    full_text = model.to_string(generated_tokens)
    
    # Handle case where to_string returns a list (when batch_size > 1)
    if isinstance(full_text, list):
        full_text = full_text[0]  # Take the first (and only) item
    
    response = full_text[len(formatted_prompt):].strip()
    
    return formatted_prompt, response

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

if __name__ == "__main__":
    # ========== CONFIGURATION - CHANGE THESE VARIABLES ==========
    model_name = 'Qwen/Qwen-1_8B-Chat'  # HookedTransformer model name
    device = 'cuda:0'  # Device for HookedTransformer
    
    dataset_path = "/mnt/SSD7/kartik/reasoning/dataset/splits/harmful_test.json"  # Path to your dataset file or HuggingFace dataset name
    question_column = "instruction"  # Column name containing questions in your dataset
    output_file = "responses.json"  # Output JSON file path
    max_new_tokens = 100  # Maximum number of new tokens to generate
    n = 100  # number of questions to generate responses for
    verbose = True  # Print detailed generation information
    # ============================================================
    
    print(f"🚀 Starting response generation with model: {model_name}")
    print(f"📊 Dataset: {dataset_path}")
    
    # Load model using HookedTransformer
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name, 
        device=device,
        trust_remote_code=True,
    )
    
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
            formatted_prompt, response = generate_response_hooked(model, question, max_new_tokens)
            
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