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

def format_prompt(prompt, apply=True):
    if apply:
        return QWEN_CHAT_TEMPLATE.format(prompt=prompt)
    else:
        return prompt

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
        return float(match.group(1))

    # Try to extract from LaTeX-style boxed answer
    match = re.search(r'\\boxed{?\$?(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))

    # As fallback, look for last number in text
    all_nums = re.findall(r'\d+(?:\.\d+)?', text)
    if all_nums:
        return float(all_nums[-1])

    return None

def instructions(thinking=True):
    instruction = (
    "Give ONLY the final numeric answer. Do NOT explain or show your work. "
    "Write nothing except the answer.\n\n"
    "Answer:"
    )
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
    n = 30
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

        # Thinking mode
        prompt_think = question + instructions(thinking=True)
        formatted_think = format_prompt(prompt_think)
        response_think = generate_response(formatted_think, max_new_tokens=max_new_tokens)
        result["response_thinking"] = response_think
        
        # Nothinking mode
        prompt_nothink = question + instructions(thinking=False)
        formatted_nothink = format_prompt(prompt_nothink)
        response_nothink = generate_response(formatted_nothink, max_new_tokens=max_new_tokens)
        result["response_nothinking"] = response_nothink

        result["predicted_thinking"] = extract_final_answer(response_think)
        result["predicted_nothinking"] = extract_final_answer(response_nothink)
        result['answer'] = final_answers[i]

        results.append(result)

    # Save results
    with open("playground/qwen3_forced_answer.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Experiment completed! Results saved to qwen3_forced_answer.json")

if __name__ == "__main__":
    main() 