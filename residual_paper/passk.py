#!/usr/bin/env python3
"""
Multiple response generation script for transformer models.

This script generates k responses for a given set of instructions using a 
transformer model (e.g., Qwen-1_8B-Chat). It supports both standard top-p 
sampling and generation with custom hooks applied to the model's residual stream.

Key Features:
- Generation of k diverse responses per instruction.
- Support for activation addition and ablation hooks.
- Efficient generation using Key-Value (KV) caching.
- Proper memory management with torch.no_grad() to prevent gradient accumulation.
- Reproducible generation with configurable random seeds.
"""
from datasets import load_dataset
import torch
import json
import random
import einops
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from torch import Tensor
import gc

# Qwen refusal tokens
QWEN_REFUSAL_TOKS = [40, 2121]  # Corresponds to ['I', 'As']

def make_actadd_hook(direction: torch.Tensor, scale: float = 1.0, device=None):
    """Create a hook to add a scaled direction to the residual stream."""
    if device is not None:
        direction = direction.to(device)
    
    def hook(resid_pre, hook):
        if device is None:
            direction_device = direction.to(resid_pre.device)
        else:
            direction_device = direction
        return resid_pre + scale * direction_device.view(1, 1, -1)
    return hook

def make_ablation_hook(direction: torch.Tensor, device=None):
    """Create a hook to ablate (remove) a direction from the residual stream."""
    if device is not None:
        direction = direction.to(device)
    
    def hook(resid_pre, hook):
        if device is None:
            direction_ = direction.to(resid_pre.device)
        else:
            direction_ = direction
        
        proj_coeff = einops.einsum(
            resid_pre, direction_, "... d_model, d_model -> ..."
        )
        proj = einops.einsum(
            proj_coeff, direction_, "..., d_model -> ... d_model"
        )
        return resid_pre - proj
    return hook

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> List[str]:
    """
    Generate text with hooks applied, re-evaluating the entire sequence for each new token.
    Supports sampling for diverse responses.
    """
    all_toks = toks.clone()

    with torch.no_grad():
        for _ in range(max_tokens_generated):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks)
            
            next_token_logits = logits[:, -1, :]
            
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(toks.shape[0]):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)
            
            all_toks = torch.cat([all_toks, next_token], dim=1)

    full_texts = model.to_string(all_toks)
    prompt_texts = model.to_string(toks)

    completions = [
        full[len(prompt):].strip()
        for full, prompt in zip(full_texts, prompt_texts)
    ]
    return completions

def get_last_token_logits(
    model,
    tokens: Int[Tensor, "batch seq"],
    fwd_hooks = [],
    batch_size: int = 16
) -> Float[Tensor, "n_instructions d_vocab"]:
    """Get logits for the last token with hooks applied."""
    all_logits = []

    with torch.no_grad():  # Prevent gradient computation during evaluation
        for i in range(0, tokens.shape[0], batch_size):
            toks = tokens[i:i + batch_size]
            # Run a forward pass with hooks
            with model.hooks(fwd_hooks=fwd_hooks):
                gen_logits = model(toks)  # shape: [batch, seq, d_vocab]
            last_logits = gen_logits[:, -1, :]  # take logits for the next token

            all_logits.append(last_logits)

    return torch.cat(all_logits, dim=0)  # [n_instructions, d_vocab]

def evaluate_refusal_with_hooks(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_fn,
    avg_direction: torch.Tensor,
    tokens_to_consider: int = 5,
    test_indices: List[int] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """Evaluate refusal behavior with hooks applied."""
    if test_indices is None:
        test_indices = list(range(min(16, len(instructions))))
    
    positions = range(tokens_to_consider)
    refusal_matrix = torch.zeros((model.cfg.n_layers, tokens_to_consider), dtype=torch.int)
    
    with torch.no_grad():
        for layer in tqdm(range(model.cfg.n_layers), desc="Layers"):
            for pos in positions:
                direction = avg_direction[layer, pos] 
                direction = direction / direction.norm()
                hook_fn_add = make_actadd_hook(direction, scale=scale, device=model.cfg.device)
                fwd_hooks_add = [(f"blocks.{layer}.hook_resid_pre", hook_fn_add)]
                
                for idx in test_indices:
                    model.reset_hooks()
                    if idx < len(instructions):
                        prompt = instructions[idx]
                        tokens = tokenize_fn([prompt])
                        logits = get_last_token_logits(model, tokens, fwd_hooks=fwd_hooks_add)
                        greedy_tokens = logits.argmax(dim=-1)
                        refusal_matrix[layer, pos] += int(greedy_tokens.item() in QWEN_REFUSAL_TOKS)
    
    return refusal_matrix

def evaluate_bypass_with_hooks(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_fn,
    avg_direction: torch.Tensor,
    tokens_to_consider: int = 5,
    test_indices: List[int] = None
) -> torch.Tensor:
    """Evaluate bypass behavior (ablation) with hooks applied."""
    if test_indices is None:
        test_indices = list(range(min(16, len(instructions))))
    
    positions = range(tokens_to_consider)
    bypass_matrix = torch.zeros((model.cfg.n_layers, tokens_to_consider), dtype=torch.int)
    
    with torch.no_grad():
        for layer in tqdm(range(model.cfg.n_layers), desc="Layers"):
            for pos in positions:
                direction = avg_direction[layer, pos] 
                direction = direction / direction.norm()
                hook_fn_ablation = make_ablation_hook(direction, device=model.cfg.device)
                fwd_hooks_ablation = [
                    (f"blocks.{layer}.hook_resid_pre", hook_fn_ablation),
                    (f"blocks.{layer}.hook_resid_mid", hook_fn_ablation),
                    (f"blocks.{layer}.hook_resid_post", hook_fn_ablation)
                ]
                
                for idx in test_indices:
                    model.reset_hooks()
                    if idx < len(instructions):
                        prompt = instructions[idx]
                        tokens = tokenize_fn([prompt])
                        logits = get_last_token_logits(model, tokens, fwd_hooks=fwd_hooks_ablation)
                        greedy_tokens = logits.argmax(dim=-1)
                        bypass_matrix[layer, pos] += int(greedy_tokens.item() in QWEN_REFUSAL_TOKS)
    
    return bypass_matrix

class MultiResponseGenerator:
    def __init__(self, model_name: str = 'Qwen/Qwen-1_8B-Chat', device: str = 'cuda', verbose: bool = True):
        """Initialize the generator with the specified model."""
        self.model_name = model_name
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(
            model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        # Chat template for Qwen
        self.chat_template = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
        
    def get_prompt(self, instruction: str) -> str:
        """Format instruction with chat template."""
        return self.chat_template.format(instruction=instruction)
    
    def get_prompt_tokens(self, instruction: str) -> torch.Tensor:
        """Tokenize the formatted prompt."""
        return self.model.to_tokens(self.get_prompt(instruction))
    
    def tokenize_chat(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize multiple prompts with chat formatting."""
        formatted = [self.get_prompt(p) for p in prompts]
        return self.model.to_tokens(formatted)
    
    def generate_k_responses_with_hooks(self, instruction: str, k: int = 5, max_tokens: int = 64, 
                                       fwd_hooks = [], seed: int = 42, instruction_id: int = 0,
                                       temperature: float = 0.7, top_p: float = 0.9, 
                                       do_sample: bool = True) -> List[str]:
        """Generate k responses for a single instruction using hooks and sampling."""
        tokens = self.get_prompt_tokens(instruction)
        responses = []
        
        if self.verbose:
            print(f"\nInstruction: {instruction}")
            if do_sample:
                print(f"Generating {k} responses with hooks (temperature={temperature}, top_p={top_p})...")
            else:
                print(f"Generating {k} responses with hooks (greedy decoding)...")
        
        for i in range(k):
            try:
                # Set seed for reproducible variability
                response_seed = seed + instruction_id * 1000 + i
                torch.manual_seed(response_seed)
                random.seed(response_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(response_seed)
                
                # Use the hooks-based generation with sampling
                completions = _generate_with_hooks(
                    self.model,
                    tokens,
                    max_tokens_generated=max_tokens,
                    fwd_hooks=fwd_hooks,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                response = completions[0] if completions else ""
                responses.append(response.strip())
                
                if self.verbose:
                    print(f"Response {i+1}: {response.strip()}")
                
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"Error generating response {i+1} for instruction: {e}")
                responses.append(error_msg)
                
                if self.verbose:
                    print(f"Response {i+1}: {error_msg}")
        
        return responses
    
    def generate_k_responses(self, instruction: str, k: int = 5, max_tokens: int = 64, 
                           top_p: float = 0.9, temperature: float = 0.7, fwd_hooks = [], 
                           seed: int = 42, instruction_id: int = 0) -> List[str]:
        """Generate k responses for a single instruction."""
        
        # If hooks are provided, use hook-based generation (with sampling)
        if fwd_hooks:
            return self.generate_k_responses_with_hooks(
                instruction, k, max_tokens, fwd_hooks, seed, instruction_id,
                temperature=temperature, top_p=top_p, do_sample=True
            )
        
        # Otherwise, use standard sampling for diverse responses.
        tokens = self.get_prompt_tokens(instruction)
        responses = []
        
        if self.verbose:
            print(f"\nInstruction: {instruction}")
            print(f"Generating {k} responses with sampling (temp={temperature}, top_p={top_p})...")
        
        for i in range(k):
            try:
                response_seed = seed + instruction_id * 1000 + i
                torch.manual_seed(response_seed)
                random.seed(response_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(response_seed)
                
                with torch.no_grad():
                    generated = self.model.generate(
                        tokens,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
                
                prompt_length = tokens.shape[-1]
                generated_tokens = generated[0, prompt_length:]
                response = self.model.to_string(generated_tokens)
                responses.append(response.strip())
                
                if self.verbose:
                    print(f"Response {i+1}: {response.strip()}")
                
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"Error generating response {i+1} for instruction: {e}")
                responses.append(error_msg)
                
                if self.verbose:
                    print(f"Response {i+1}: {error_msg}")
        
        return responses

    def process_dataset(self, instructions: List[str], dataset_path: str, k: int = 5, 
                       max_tokens: int = 64, top_p: float = 0.9, temperature: float = 0.7,
                       fwd_hooks = [], seed: int = 42, greedy_when_hooked: bool = False) -> Dict[str, Any]:
        """Process a dataset and generate k responses for each instruction."""
        results = {
            'responses': [],
            'metadata': {
                'model': self.model_name,
                'dataset_path': dataset_path,
                'num_instructions': len(instructions),
                'k_responses_per_instruction': k,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'temperature': temperature,
                'seed': seed,
                'hooks_applied': len(fwd_hooks) > 0,
                'num_hooks': len(fwd_hooks),
                'greedy_when_hooked': greedy_when_hooked,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if fwd_hooks:
            if greedy_when_hooked:
                print(f"Using {len(fwd_hooks)} forward hooks with greedy decoding")
            else:
                print(f"Using {len(fwd_hooks)} forward hooks with sampling (temp={temperature}, top_p={top_p})")
        else:
            print(f"Generating {k} responses for {len(instructions)} instructions from {dataset_path}...")

        for i, instruction in enumerate(tqdm(instructions, desc=f"Processing {dataset_path.split('/')[-1] if dataset_path else 'dataset'}")):
            if fwd_hooks and greedy_when_hooked:
                responses = self.generate_k_responses_with_hooks(
                    instruction, k, max_tokens, fwd_hooks, seed, i,
                    temperature=1.0, top_p=1.0, do_sample=False
                )
            else:
                responses = self.generate_k_responses(
                    instruction=instruction,
                    k=k,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    fwd_hooks=fwd_hooks,
                    seed=seed,
                    instruction_id=i
                )
            
            responses_dict = {}
            for j, response in enumerate(responses):
                responses_dict[str(j)] = response
            
            instruction_result = {
                'instruction_id': i,
                'instruction': instruction,
                'responses': responses_dict
            }
            
            results['responses'].append(instruction_result)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")

def load_dataset_json(path: str) -> List[str]:
    """Load dataset and return list of instructions."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [item['instruction'] for item in data]

def create_hooks_from_direction(avg_direction: torch.Tensor, layer: int, pos: int, 
                               hook_type: str = "actadd", scale: float = 1.0, device=None) -> List:
    """Create hooks from avg_direction data."""
    direction = avg_direction[layer, pos]
    direction = direction / direction.norm()
    
    if hook_type == "actadd":
        hook_fn = make_actadd_hook(direction, scale=scale, device=device)
        return [(f"blocks.{layer}.hook_resid_pre", hook_fn)]
    
    elif hook_type == "ablation":
        hook_fn = make_ablation_hook(direction, device=device)
        return [
            (f"blocks.{layer}.hook_resid_pre", hook_fn),
            (f"blocks.{layer}.hook_resid_mid", hook_fn),
            (f"blocks.{layer}.hook_resid_post", hook_fn)
        ]
    
    else:
        raise ValueError(f"Unknown hook_type: {hook_type}. Use 'actadd' or 'ablation'")

if __name__ == "__main__":

    # Configuration variables
    model_name = 'Qwen/Qwen-1_8B-Chat'
    device = 'cuda'
    k = 5  # number of responses per instruction
    max_tokens = 100
    top_p = 0.9
    temperature = 0.7
    n = 400  # number of instructions to process from dataset
    seed = 42
    output_dir = 'residual_paper/k_responses_results'
    verbose = True


    dataset_path = "walledai/XSTest"
    dataset = load_dataset(dataset_path)
    dataset_name = dataset_path.split('/')[-1]
    instructions = dataset['test']['prompt']
    # # Data path
    # dataset_path = '/mnt/SSD7/kartik/reasoning/dataset/processed/advbench.json'
    # dataset_name = dataset_path.split('/')[-1].split('.')[0]
    # instructions = load_dataset_json(dataset_path)
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Initialize generator
    generator = MultiResponseGenerator(model_name=model_name, device=device, verbose=verbose)

    # Load dataset
    print("Loading dataset...")
    

    print(f"Loaded {len(instructions)} instructions")

    # Sample n instructions from dataset
    sample_instructions = random.sample(instructions, min(n, len(instructions)))

    print(f"Processing {len(sample_instructions)} instructions")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define hooks
    fwd_hooks = []
    # NOTE: When hooks are applied, the generation now uses proper sampling (temperature/top_p)
    # instead of greedy decoding, so different seeds will produce different responses.
    avg_direction = torch.load('/mnt/SSD7/kartik/reasoning/residual_paper/avg_direction.pt')  # Load your direction data
    layer = 11
    pos = 4
    scale = 10.0
    # fwd_hooks = create_hooks_from_direction(avg_direction, layer=layer, pos=pos, hook_type="actadd", scale=scale, device=generator.model.cfg.device)
    # fwd_hooks = create_hooks_from_direction(avg_direction, layer=layer, pos=pos, hook_type="ablation", device=generator.model.cfg.device)

    # Process dataset
    results = generator.process_dataset(
        instructions=sample_instructions,
        dataset_path=dataset_path,
        k=k,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        fwd_hooks=fwd_hooks,
        seed=seed,
        greedy_when_hooked=False  # Set to True if you want deterministic generation
    )

    # Save results
    filename = f"evaluation_passk_{dataset_name}.json"
    generator.save_results(results, output_dir / filename)

    print(f"\nGeneration complete!")
    print(f"Results: {output_dir / filename}")
    
    # Example of how to run refusal evaluation with hooks (commented out):
    # if avg_direction is loaded:
    # refusal_matrix = evaluate_refusal_with_hooks(
    #     generator.model, 
    #     sample_instructions, 
    #     generator.tokenize_chat, 
    #     avg_direction,
    #     tokens_to_consider=5
    # )
    # print("Refusal matrix shape:", refusal_matrix.shape)
    # torch.save(refusal_matrix, output_dir / f"refusal_matrix_{timestamp}.pt")
