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

def load_dataset_questions(dataset_path=None, dataset_name=None, n_samples=None):
    """Load questions from various dataset sources
    
    Args:
        dataset_path (str): Path to local JSON file (for harmful datasets)
        dataset_name (str): HuggingFace dataset name 
        n_samples (int): Number of samples to take
    
    Returns:
        list: List of question strings
    """
    questions = []
    
    if dataset_path:
        # Load from local JSON file (harmful datasets)
        print(f"Loading dataset from path: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract questions (assumes format like harmful datasets)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'instruction' in data[0]:
                questions = [item['instruction'] for item in data]
            elif isinstance(data[0], str):
                questions = data
        
        print(f"Loaded {len(questions)} questions from {dataset_path}")
        
    elif dataset_name:
        # Load from HuggingFace datasets
        from datasets import load_dataset
        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name)
        
        # Extract questions (adjust key based on dataset)
        if 'Question' in ds['train'][0]:
            questions = [item['Question'] for item in ds['train']]
        elif 'question' in ds['train'][0]:
            questions = [item['question'] for item in ds['train']]
        else:
            # Print available keys to help debug
            print(f"Available keys: {list(ds['train'][0].keys())}")
            raise ValueError("Could not find question field in dataset")
        
        print(f"Loaded {len(questions)} questions from {dataset_name}")
    
    # Sample if requested
    if n_samples and n_samples < len(questions):
        random.seed(0)
        questions = random.sample(questions, n_samples)
        print(f"Sampled {len(questions)} questions")
    
    return questions

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

def save_residuals_to_pt(residuals_data: dict, model_name: str, title: str = "residuals", filename: str = None, output_folder: str = ".") -> str:
    """Save residuals tensor data to a .pt file"""
    # Clean model name for folder (replace / with _)
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    
    # Create model-specific folder inside output folder
    model_folder = os.path.join(output_folder, clean_model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    if filename is None:
        filename = f"{title}.pt"
    
    # Save to model folder
    filepath = os.path.join(model_folder, filename)
    
    # Save tensor data
    torch.save(residuals_data, filepath)
    
    print(f"Residuals saved to: {filepath}")
    return filepath

def process_mode(questions, model, tokenizer, residuals, think_mode=True, n=100, verbose=True):
    """Process questions in specified mode and capture residuals
    
    Args:
        questions (list): List of question strings to process
        think_mode (bool): If True, process in THINK mode. If False, process in NOTHINK mode.
    """
    mode_str = "THINK" if think_mode else "NOTHINK"
    print(f"Processing {min(n, len(questions))} questions in {mode_str} mode...")
    print("=" * 60)
    
    # Get EOI tokens - these are the tokens after {prompt} in the template
    eoi_tokens = _get_eoi_toks(tokenizer)
    if verbose:
        print(f"EOI tokens: {eoi_tokens} (length: {len(eoi_tokens)})")
    
    # Get model dimensions
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    n_eoi_tokens = len(eoi_tokens)
    actual_n = min(n, len(questions))
    
    # Pre-allocate tensors for all prompts
    # Shape: (n_prompts, n_layers, n_eoi_tokens, hidden_size)
    pre_residuals = torch.zeros(actual_n, num_layers, n_eoi_tokens, hidden_size)
    post_residuals = torch.zeros(actual_n, num_layers, n_eoi_tokens, hidden_size)
    
    for i in range(actual_n):
        question = questions[i]
        question_id = i
        
        if verbose and i % 10 == 0:
            print(f"Processing Question {question_id} ({mode_str} mode)")
        
        # Clear residuals for this question
        residuals.clear()
        
        # Format prompt based on mode
        if think_mode:
            formatted_prompt = format_prompt(question)
        else:
            nothink_question = question + " /nothink"
            formatted_prompt = format_prompt(nothink_question)
            
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Run model to capture residuals
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract EOI token residuals and store directly in pre-allocated tensors
        for layer_idx in range(num_layers):
            if layer_idx in residuals:
                # Get the last n_eoi_tokens positions for EOI tokens
                if "pre" in residuals[layer_idx]:
                    pre_tensor = residuals[layer_idx]["pre"]
                    if pre_tensor.size(1) >= n_eoi_tokens:
                        pre_residuals[i, layer_idx] = pre_tensor[0, -n_eoi_tokens:, :].cpu()
                
                if "post" in residuals[layer_idx]:
                    post_tensor = residuals[layer_idx]["post"]
                    if post_tensor.size(1) >= n_eoi_tokens:
                        post_residuals[i, layer_idx] = post_tensor[0, -n_eoi_tokens:, :].cpu()
    
    print(f"Completed {mode_str} mode processing for {actual_n} questions")
    
    # Create residuals data dict in the same format as generate_resp.py
    residuals_data = {
        'pre': pre_residuals,
        'post': post_residuals,
        'metadata': {
            'n_prompts': actual_n,
            'n_layers': num_layers,
            'n_eoi_tokens': n_eoi_tokens,
            'd_model': hidden_size,
            'eoi_token_ids': eoi_tokens,
            'questions': questions[:actual_n],
            'mode': mode_str,
            'shape_description': '(n_prompts, n_layers, n_eoi_tokens, d_model)'
        }
    }
    
    return residuals_data

if __name__ == "__main__":
    # Configuration
    n = 100  # Number of questions to process
    verbose = True
    output_folder_residuals = 'residuals/general_knowledge'
    model_name = "Qwen/Qwen3-4B"
    dataset_name = "MuskumPillerum/General-Knowledge"
    
    # Load questions from dataset using the same pattern as generate_resp.py
    questions = load_dataset_questions(dataset_name=dataset_name, n_samples=None)
    
    # Create random indices for sampling (same pattern as generate_resp.py)
    random.seed(0)
    random_indices = random.sample(range(len(questions)), n)
    
    # Extract questions using random indices
    sampled_questions = [questions[idx] for idx in random_indices]
    
    print(f"Dataset loaded with {len(questions)} total questions")
    print(f"Sampled {len(sampled_questions)} questions using random seed 0")
    print("=" * 60)
    print("Running Residual Capture Analysis")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Setup hooks
    residuals = setup_hooks(model)
    
    # Create residuals directory if it doesn't exist
    os.makedirs(output_folder_residuals, exist_ok=True)
    
    # Process THINK mode
    print("\nStep 1: Processing THINK mode...")
    think_residuals_data = process_mode(
        sampled_questions, model, tokenizer, residuals, 
        think_mode=True, n=n, verbose=verbose
    )
    
    # Save think residuals
    think_filepath = save_residuals_to_pt(
        think_residuals_data, 
        model_name=model_name, 
        title="think_residuals", 
        output_folder=output_folder_residuals
    )
    
    # Process NOTHINK mode
    print("\nStep 2: Processing NOTHINK mode...")
    nothink_residuals_data = process_mode(
        sampled_questions, model, tokenizer, residuals, 
        think_mode=False, n=n, verbose=verbose
    )
    
    # Save nothink residuals
    nothink_filepath = save_residuals_to_pt(
        nothink_residuals_data, 
        model_name=model_name, 
        title="nothink_residuals", 
        output_folder=output_folder_residuals
    )
    
    if think_residuals_data is not None and nothink_residuals_data is not None:
        print("\nExperiment completed successfully!")
        print(f"THINK residuals saved to: {think_filepath}")
        print(f"NOTHINK residuals saved to: {nothink_filepath}")
        print(f"Total questions processed: {think_residuals_data['pre'].size(0)}")
        print(f"Shape: [n_prompts={think_residuals_data['pre'].size(0)}, n_layers={think_residuals_data['pre'].size(1)}, n_eoi_tokens={think_residuals_data['pre'].size(2)}, d_model={think_residuals_data['pre'].size(3)}]")
    else:
        print("Experiment failed. Please check your setup.")