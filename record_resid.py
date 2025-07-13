import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import random
import re
from tqdm import tqdm
from collections import defaultdict

# Chat template for Qwen
QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

def format_prompt(prompt):
    """Format prompt with Qwen chat template"""
    return QWEN_CHAT_TEMPLATE.format(prompt=prompt)

def load_harmful_dataset(dataset_path="playground/dataset/splits/harmful_test.json"):
    """Load harmful questions dataset"""
    print(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        harmful = json.load(f)
    
    print(f"Loaded {len(harmful)} harmful questions")
    return harmful

def get_model_filename(model_name):
    """Convert model name to a safe filename"""
    # Replace slashes and other problematic characters
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe_name

def load_model_and_tokenizer(model_name):
    """Load tokenizer and model"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def _get_eoi_toks(tokenizer):
    """Get end-of-instruction tokens"""
    return tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{prompt}")[-1])

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

def run_residual_experiment(harmful_questions, model, tokenizer, model_name, residuals, n=100, verbose=True):
    """
    Run residual capture experiment: generate both think and nothink responses,
    capture residuals for all layers, and extract EOI tokens
    EOI tokens are End-of-Instruction tokens: the tokens after user prompt before assistant response
    """
    print(f"Running residual experiment with {n} questions...")
    print("=" * 60)
    
    # Get EOI tokens - these are the tokens after {prompt} in the template
    eoi_tokens = _get_eoi_toks(tokenizer)
    print(f"EOI tokens: {eoi_tokens} (length: {len(eoi_tokens)})")
    
    # Get model dimensions
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    
    # Step 1: Process think mode
    print("\nStep 1: Processing THINK mode...")
    think_pre_residuals = []  # List to store pre residuals for each question
    think_post_residuals = []  # List to store post residuals for each question
    
    for i in range(min(n, len(harmful_questions))):
        question = harmful_questions[i]['instruction']
        question_id = i + 1
        
        if verbose and i % 10 == 0:
            print(f"Processing Question {question_id} (THINK mode)")
        
        # Clear residuals for this question
        residuals.clear()
        
        # Format prompt for think mode
        formatted_prompt = format_prompt(question)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Run model to capture residuals
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract EOI token residuals and stack into matrices
        # Shape: [layers, len(eoi_tokens), hidden_size]
        pre_matrix = torch.zeros(num_layers, len(eoi_tokens), hidden_size)
        post_matrix = torch.zeros(num_layers, len(eoi_tokens), hidden_size)
        
        for layer_idx in range(num_layers):
            if layer_idx in residuals:
                # Get the last len(eoi_tokens) positions for EOI tokens
                if "pre" in residuals[layer_idx]:
                    pre_tensor = residuals[layer_idx]["pre"]
                    if pre_tensor.size(1) >= len(eoi_tokens):
                        pre_matrix[layer_idx] = pre_tensor[0, -len(eoi_tokens):, :].cpu()
                
                if "post" in residuals[layer_idx]:
                    post_tensor = residuals[layer_idx]["post"]
                    if post_tensor.size(1) >= len(eoi_tokens):
                        post_matrix[layer_idx] = post_tensor[0, -len(eoi_tokens):, :].cpu()
        
        think_pre_residuals.append(pre_matrix)
        think_post_residuals.append(post_matrix)
    
    print(f"Completed THINK mode processing for {len(think_pre_residuals)} questions")
    
    # Step 2: Process nothink mode
    print("\nStep 2: Processing NOTHINK mode...")
    nothink_pre_residuals = []  # List to store pre residuals for each question
    nothink_post_residuals = []  # List to store post residuals for each question
    
    for i in range(min(n, len(harmful_questions))):
        question = harmful_questions[i]['instruction']
        question_id = i + 1
        
        if verbose and i % 10 == 0:
            print(f"Processing Question {question_id} (NOTHINK mode)")
        
        # Clear residuals for this question
        residuals.clear()
        
        # Format prompt for nothink mode
        nothink_question = question + " /nothink"
        formatted_prompt = format_prompt(nothink_question)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Run model to capture residuals
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract EOI token residuals and stack into matrices
        # Shape: [layers, len(eoi_tokens), hidden_size]
        pre_matrix = torch.zeros(num_layers, len(eoi_tokens), hidden_size)
        post_matrix = torch.zeros(num_layers, len(eoi_tokens), hidden_size)
        
        for layer_idx in range(num_layers):
            if layer_idx in residuals:
                # Get the last len(eoi_tokens) positions for EOI tokens
                if "pre" in residuals[layer_idx]:
                    pre_tensor = residuals[layer_idx]["pre"]
                    if pre_tensor.size(1) >= len(eoi_tokens):
                        pre_matrix[layer_idx] = pre_tensor[0, -len(eoi_tokens):, :].cpu()
                
                if "post" in residuals[layer_idx]:
                    post_tensor = residuals[layer_idx]["post"]
                    if post_tensor.size(1) >= len(eoi_tokens):
                        post_matrix[layer_idx] = post_tensor[0, -len(eoi_tokens):, :].cpu()
        
        nothink_pre_residuals.append(pre_matrix)
        nothink_post_residuals.append(post_matrix)
    
    print(f"Completed NOTHINK mode processing for {len(nothink_pre_residuals)} questions")
    
    # Step 3: Save results
    print("\nStep 3: Saving results...")
    
    # Create model-specific filename
    model_filename = get_model_filename(model_name)
    
    # Stack all questions into single tensors
    # Shape: [questions, layers, len(eoi_tokens), hidden_size]
    think_pre_stacked = torch.stack(think_pre_residuals, dim=0)
    think_post_stacked = torch.stack(think_post_residuals, dim=0)
    nothink_pre_stacked = torch.stack(nothink_pre_residuals, dim=0)
    nothink_post_stacked = torch.stack(nothink_post_residuals, dim=0)
    
    # Save think residuals
    think_output_file = f'residuals/residuals_think_{model_filename}.pt'
    torch.save({
        'pre': think_pre_stacked,
        'post': think_post_stacked,
        'eoi_token_ids': eoi_tokens,
        'questions': [harmful_questions[i]['instruction'] for i in range(min(n, len(harmful_questions)))]
    }, think_output_file)
    
    # Save nothink residuals
    nothink_output_file = f'residuals/residuals_nothink_{model_filename}.pt'
    torch.save({
        'pre': nothink_pre_stacked,
        'post': nothink_post_stacked,
        'eoi_token_ids': eoi_tokens,
        'questions': [harmful_questions[i]['instruction'] for i in range(min(n, len(harmful_questions)))]
    }, nothink_output_file)
    
    print(f"THINK residuals saved to: {think_output_file}")
    print(f"NOTHINK residuals saved to: {nothink_output_file}")
    print(f"Residual shapes: {think_pre_stacked.shape} [questions, layers, eoi_tokens, hidden_size]")
    
    return think_pre_stacked, think_post_stacked, nothink_pre_stacked, nothink_post_stacked

if __name__ == "__main__":
    # Configuration
    n = 100  # Number of questions to process
    dataset_path = "playground/dataset/splits/harmful_test.json"
    verbose = True

    # model_name = "Qwen/Qwen3-1.7B"
    model_name = "Qwen/Qwen3-4B"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("=" * 60)
    print("Running Residual Capture Analysis")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Setup hooks
    residuals = setup_hooks(model)
    
    # Create residuals directory if it doesn't exist
    os.makedirs("residuals", exist_ok=True)
    
    # Load harmful dataset
    harmful_questions = load_harmful_dataset(dataset_path)
    
    # Run residual experiment
    think_pre, think_post, nothink_pre, nothink_post = run_residual_experiment(
        harmful_questions, model, tokenizer, model_name, residuals,
        n=n, verbose=verbose
    )
    
    if think_pre is not None and think_post is not None and nothink_pre is not None and nothink_post is not None:
        print("\nExperiment completed successfully!")
        model_filename = get_model_filename(model_name)
        print(f"THINK residuals saved to: residuals/residuals_think_{model_filename}.pt")
        print(f"NOTHINK residuals saved to: residuals/residuals_nothink_{model_filename}.pt")
        print(f"Total questions processed: {think_pre.size(0)}")
        print(f"Shape: [questions={think_pre.size(0)}, layers={think_pre.size(1)}, eoi_tokens={think_pre.size(2)}, hidden_size={think_pre.size(3)}]")
    else:
        print("Experiment failed. Please check your setup.")