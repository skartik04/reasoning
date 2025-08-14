#!/usr/bin/env python3
"""
Temporary script to run gaslight_generation_chat_completion.py with all 3 models sequentially.
This script will run overnight and handle errors gracefully.
"""

import subprocess
import sys
import time
from datetime import datetime
import os

def log_message(message):
    """Print message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_model(model_name, script_path):
    """Run gaslight_generation_chat_completion.py with a specific model."""
    log_message(f"Starting generation for model: {model_name}")
    
    try:
        # Read the original script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Replace the model_name line with the current model
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('model_name = ') and not line.strip().startswith('#'):
                lines[i] = f"    model_name = '{model_name}'"
                break
        
        # Write temporary script
        temp_script = script_path.replace('.py', '_temp.py')
        with open(temp_script, 'w') as f:
            f.write('\n'.join(lines))
        
        # Run the temporary script with real-time output
        start_time = time.time()
        timeout_seconds = 7200  # 2 hours
        
        # Use subprocess.Popen for real-time output streaming
        process = subprocess.Popen([sys.executable, temp_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)
        
        # Stream output in real-time with timeout check
        while True:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                log_message(f"⏰ Timeout reached for {model_name}, terminating...")
                process.terminate()
                process.wait(timeout=10)  # Wait up to 10 seconds for graceful termination
                if process.poll() is None:  # If still running, force kill
                    process.kill()
                    process.wait()
                raise subprocess.TimeoutExpired(temp_script, timeout_seconds)
            
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Print the original script's output
        
        # Wait for process to complete and get return code
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            log_message(f"✅ Successfully completed {model_name} in {elapsed_time:.1f} seconds")
        else:
            log_message(f"❌ Error running {model_name} with return code: {return_code}")
        
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)
            
        return return_code == 0
        
    except subprocess.TimeoutExpired:
        log_message(f"⏰ Timeout expired for {model_name} after 2 hours")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        log_message(f"💥 Exception while running {model_name}: {str(e)}")
        return False

def main():
    """Main function to run all models sequentially."""
    script_path = '/mnt/SSD7/kartik/reasoning/gaslight_generation.py'
    
    models = [
        'meta-llama/Llama-3.1-8B-Instruct',
        'Qwen/Qwen3-4B-Instruct-2507'
    ]
    
    log_message("🚀 Starting overnight gaslighting chat completion generation for all models")
    log_message(f"Models to process: {models}")
    
    start_time = datetime.now()
    results = {}
    
    for i, model in enumerate(models, 1):
        log_message(f"📊 Processing model {i}/{len(models)}: {model}")
        
        try:
            success = run_model(model, script_path)
            results[model] = "SUCCESS" if success else "FAILED"
            
            # Add a small delay between models to allow system cleanup
            if i < len(models):  # Don't sleep after the last model
                log_message("😴 Waiting 30 seconds before next model...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            log_message("⚠️  Interrupted by user")
            results[model] = "INTERRUPTED"
            break
        except Exception as e:
            log_message(f"💥 Unexpected error with {model}: {str(e)}")
            results[model] = "ERROR"
            continue
    
    # Final summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    log_message("🏁 All models processing complete!")
    log_message(f"Total execution time: {total_time}")
    log_message("📋 Results summary:")
    
    for model, status in results.items():
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        log_message(f"  {status_emoji} {model}: {status}")
    
    successful_models = sum(1 for status in results.values() if status == "SUCCESS")
    log_message(f"🎯 Success rate: {successful_models}/{len(results)} models completed successfully")

if __name__ == "__main__":
    main()
