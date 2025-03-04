import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import sys

def setup_model():
    """
    Set up the model and tokenizer, with fallback options.
    Returns the model and tokenizer.
    """
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        print(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    
    # Try to load DeepSeek model first
    try:
        print("Using a smaller model for testing...")
        model_name = "facebook/opt-125m"  # ~500MB instead of 10GB
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings for memory constraints
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"  # Let the library decide how to allocate across GPUs or CPU
        )
        print("Successfully loaded DeepSeek model!")
        
    except Exception as e:
        print(f"Failed to load DeepSeek model: {e}")
        print("Falling back to StarCoder model...")
        
        model_name = "bigcode/starcoder"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Successfully loaded StarCoder model!")
    
    return model, tokenizer

def generate_code(prompt: str, model, tokenizer) -> str:
    """
    Generate code based on the given prompt.
    
    Args:
        prompt: The input prompt for code generation
        model: The loaded model
        tokenizer: The loaded tokenizer
        
    Returns:
        The generated code as a string
    """
    # Format the prompt appropriately
    if "deepseek" in tokenizer.name_or_path:
        # DeepSeek format
        formatted_prompt = f"<｜begin▁of▁conversation｜>\n<｜human｜>\n{prompt}\n<｜assistant｜>"
    else:
        # StarCoder format
        formatted_prompt = f"<human>: {prompt}\n<bot>:"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    print("Generating response...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the model's response
    if "deepseek" in tokenizer.name_or_path:
        response = generated_text.split("<｜assistant｜>")[-1].strip()
    else:
        response = generated_text.split("<bot>:")[-1].strip()
    
    return response

def main():
    """Main function to demonstrate the code generation."""
    print("Setting up the model...")
    model, tokenizer = setup_model()
    
    # Get prompt from command line arguments or use default
    if len(sys.argv) > 1:
        example_prompt = sys.argv[1]
    else:
        example_prompt = "Write a Python function to reverse a string."
    
    print("\n" + "="*50)
    print(f"PROMPT: {example_prompt}")
    print("="*50)
    
    # Generate and print the response
    response = generate_code(example_prompt, model, tokenizer)
    
    print("\n" + "="*50)
    print("GENERATED CODE:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    main()