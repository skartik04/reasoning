#!/usr/bin/env python3
"""
GPT-4o Harmful Response Generator Script
Generates responses to harmful questions and responses using GPT-4o using HarmBench dataset.
"""

import json
import os
import time
from dotenv import load_dotenv
import openai
from datasets import load_dataset
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv('/home/kartik/all_keys/.env')

def generate_response_gpt4o(prompt, client, max_tokens=700):
    """Generate response to a given prompt using GPT-4o"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        content = response.choices[0].message.content
        return content.strip() if content else "ERROR: No content returned"
    except Exception as e:
        print(f"Error generating response: {e}")
        return "ERROR"

def generate_gpt4o_responses(prompts, client, n=100, verbose=True, max_tokens=700):
    """
    Generic function to process any list of prompts with GPT-4o
    """
    print(f"Processing {min(n, len(prompts))} prompts with GPT-4o...")
    print("=" * 60)
    
    responses_data = []
    
    for i in range(min(n, len(prompts))):
        prompt = prompts[i]
        prompt_id = i + 1
        
        if verbose and i % 10 == 0:
            print(f"Processing Prompt {prompt_id}")
        
        # Generate GPT-4o response
        gpt4o_response = generate_response_gpt4o(prompt, client, max_tokens=max_tokens)
        
        if verbose:
            print(f"\nPrompt {prompt_id}:")
            print(f"Input: {prompt[:200]}..." if len(prompt) > 200 else prompt)
            print(f"GPT-4o: {gpt4o_response[:200]}..." if len(gpt4o_response) > 200 else gpt4o_response)
            print("-" * 60)
        
        responses_data.append({
            "id": prompt_id,
            "prompt": prompt,
            "gpt4o_response": gpt4o_response
        })
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Completed processing {len(responses_data)} prompts with GPT-4o")
    return responses_data

def save_data(data, filename, output_folder="responses", model_name="gpt4o_completions"):
    """Save data to JSON file in organized folder structure"""
    # Create folder structure: responses/gpt4o_completions/
    model_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # Save to model folder
    filepath = os.path.join(model_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Configuration
    n = 100  # Number of samples to process
    verbose = True
    max_tokens = 100
    dataset_name = "walledai/HarmBench"
    dataset_config = "contextual"
    output_folder = "responses"
    model_name = "gpt4o_completions"
    
    print("=" * 60)
    print("Running Complete GPT-4o Analysis on HarmBench Dataset")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully!")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        exit(1)
    
    # Load HarmBench dataset
    print(f"\nLoading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, dataset_config)
    
    # Extract questions and responses from dataset
    questions = []
    responses = []
    for prompt_data in ds['train']:
        questions.append(prompt_data['prompt'])
        responses.append(prompt_data['context'])
    
    print(f"Dataset loaded with {len(questions)} questions and {len(responses)} responses")
    print(f"Processing {n} samples...")
    print(f"Output folder: {output_folder}/{model_name}/")
    
    # Step 1: Process questions with GPT-4o
    print("\nStep 1: Processing questions with GPT-4o...")
    questions_data = generate_gpt4o_responses(
        questions, client, n=n, verbose=verbose, max_tokens=max_tokens
    )

    questions_file = save_data(questions_data, 'questions.json', output_folder, model_name)

    # Step 2: Process responses with GPT-4o  
    print("\nStep 2: Processing responses with GPT-4o...")
    responses_data = generate_gpt4o_responses(
        responses, client, n=n, verbose=verbose, max_tokens=max_tokens
    )
    
    responses_file = save_data(responses_data, 'responses.json', output_folder, model_name)
    
    if questions_data and responses_data:
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print("=" * 60)
        print(f"Questions data saved to: {questions_file}")
        print(f"Responses data saved to: {responses_file}")
        print(f"Total questions processed: {len(questions_data)}")
        print(f"Total responses processed: {len(responses_data)}")
    else:
        print("Experiment failed. Please check your setup.") 