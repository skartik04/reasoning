import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import re
from tqdm import tqdm
import json

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

print(f"Model loaded on device: {model.device}")

# Chat template
QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
QWEN_CHAT_TEMPLATE_NO = "<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_prompt(prompt):
    return QWEN_CHAT_TEMPLATE_NO.format(prompt=prompt)

def tokenize_input(text):
    """Tokenize the input text and return input_ids and attention_mask."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs

def generate_response(formatted_prompt, max_new_tokens=1000):
    inputs = tokenize_input(formatted_prompt).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def extract_final_answer(text):
    # Try strict match first: 'Answer: <number>'
    match = re.search(r'Answer:\s*\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return int(float(match.group(1)))

    # Try to extract from LaTeX-style boxed answer
    match = re.search(r'\\boxed{?\$?(\d+(?:\.\d+)?)', text)
    if match:
        return int(float(match.group(1)))

    # As fallback, look for last number in text
    all_nums = re.findall(r'\d+(?:\.\d+)?', text)
    if all_nums:
        return int(float(all_nums[-1]))

    return None

def instructions(thinking=True, concise=True):
    if concise:
        instruction = (
            "Give your reasoning step by step, but be as concise as possible. "
        )
    else:
        instruction = ""
    nothink_token = "/nothink"
    if thinking:
        return instruction
    else:
        return instruction + nothink_token

def main():
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    # Prepare data
    n = 20
    random.seed(0)
    indices = random.sample(range(len(dataset['train'])), n)
    qa_pairs = dataset['train'][indices]
    questions = qa_pairs['question']
    answers = qa_pairs['answer']
    
    final_answers = []
    for answer in answers:
        numbers = re.findall(r'\d+', answer)
        final_answers.append(int(numbers[-1]) if numbers else '')
    
    # Run experiments
    print(f"Running experiments on {n} questions...")
    results = []
    max_new_tokens = 10000
    
    for i, question in enumerate(tqdm(questions)):
        result = {
            "question": question,
        }
        
        # Nothinking mode
        prompt_nothink = question + instructions(thinking=True)
        formatted_nothink = format_prompt(prompt_nothink)
        response_nothink = generate_response(formatted_nothink, max_new_tokens=max_new_tokens)
        result["response_nothinking"] = response_nothink

        result["predicted_nothinking"] = extract_final_answer(response_nothink)
        result['answer'] = final_answers[i]

        results.append(result)

    # Save results
    with open("playground/qwen3_concise_no.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Experiment completed! Results saved to qwen3_concise_no.json")

if __name__ == "__main__":
    main() 