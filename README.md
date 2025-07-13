# Harmful Response Generator

This script generates responses to harmful questions using the Qwen model in both "think" and "nothink" modes, then analyzes the responses using GPT-4o.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a .env file:**
   Create a file named `.env` in the same directory as the script with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Prepare your dataset:**
   Make sure your harmful questions dataset is at `playground/dataset/splits/harmful_test.json`

## Usage

### Basic Usage
```bash
python harmful_response_generator.py
```

### Configuration
Edit the configuration variables in the script:
- `n`: Number of questions to process (default: 10)
- `dataset_path`: Path to your harmful questions dataset
- `model_name`: Qwen model to use (default: "Qwen/Qwen3-1.7B")
- `RUN_EXPERIMENTS`: Set to `True` to run think/nothink experiments
- `CREATE_JOINT_JSON`: Set to `True` to create joint JSON file
- `RUN_ANALYSIS`: Set to `True` to run GPT-4o analysis (requires OpenAI API key)

### Output Files
- `harmful_responses_think.json`: Responses in "think" mode
- `harmful_responses_nothink.json`: Responses in "nothink" mode
- `combined_harmful_responses.json`: Combined results
- `joint_harmful_responses.json`: Joint format for analysis
- `harmful_response_analysis.json`: GPT-4o analysis results

## Features

1. **Think/Nothink Experiments**: Generates responses with and without the `/nothink` token
2. **JSON Logging**: Saves all responses in structured JSON format
3. **GPT-4o Analysis**: Uses OpenAI's GPT-4o to analyze response harmfulness
4. **Configurable**: Easy to modify parameters and run specific steps
5. **Error Handling**: Proper error handling for missing files and API issues

## Notes

- The script uses greedy decoding for deterministic results
- Rate limiting is applied for OpenAI API calls (1 second between requests)
- Make sure you have sufficient GPU memory for the Qwen model
- The GPT-4o analysis requires a valid OpenAI API key