import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import sys

def setup_model():
    """
    Set up the model and tokenizer with CPU optimizations for the smaller DeepSeek variant.
    Returns the model and tokenizer.
    """
    device = "cpu"
    print(f"Using device: {device}")
    print(f"Available CPU cores: {os.cpu_count()}")

    try:
        print("Attempting to load DeepSeek model...")
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Smaller 1.3B model

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with CPU-specific optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 to halve memory usage (~1.3GB for weights)
            low_cpu_mem_usage=True,     # Minimize memory fragmentation
            device_map="cpu"            # Explicitly map to CPU
        )
        print("Successfully loaded DeepSeek model!")

    except Exception as e:
        print(f"Failed to load DeepSeek model: {e}")
        print("Falling back to a smaller open model...")
        model_name = "Salesforce/codegen-350M-mono"  # Alternative small model (~700MB in FP32)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        print("Successfully loaded CodeGen model!")

    # Set number of threads for CPU parallelism
    torch.set_num_threads(os.cpu_count())  # Use all available CPU cores
    return model, tokenizer

def parse_input(input_text: str) -> tuple[str, str, str]:
    """
    Parse the combined input to extract file content and user query.
    Returns (file_content, language, user_query) tuple.
    """
    if "File content" in input_text and "User query:" in input_text:
        # Extract language if specified
        language = "unknown"
        if "File content (" in input_text and ")" in input_text:
            language = input_text.split("File content (")[1].split(")")[0]

        # Split content and query
        parts = input_text.split("User query:")
        file_content = parts[0].replace(f"File content ({language}):", "").strip()
        user_query = parts[1].strip()
        return file_content, language, user_query
    else:
        # No file content, just a direct query
        return "", "", input_text

def generate_code(prompt: str, model, tokenizer) -> str:
    """
    Generate code based on the given prompt with CPU optimizations.
    Includes system instruction and better output cleaning.
    """
    # Parse the input
    file_content, language, user_query = parse_input(prompt)
    
    # System instruction to improve response quality
    system_instruction = """You are a senior software developer with extensive experience. 
Your responses should be clear, well-documented, and follow best practices. 
When explaining code, be concise but thorough. 
When writing code, include comments and error handling."""
    
    # Construct an appropriate prompt based on available context
    if file_content:
        formatted_prompt = f"""<｜begin▁of▁conversation｜>
<｜system｜>
{system_instruction}
<｜human｜>
I have the following code in {language}:

{file_content}

{user_query}
<｜assistant｜>"""
    else:
        formatted_prompt = f"""<｜begin▁of▁conversation｜>
<｜system｜>
{system_instruction}
<｜human｜>
{user_query}
<｜assistant｜>"""

    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")

    # Generate response
    print("Generating response...")
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            min_length=10
        )

    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")

    # Decode the output and clean up the response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Clean up the response by removing all special tokens and conversation markers
    response = (full_response
        .split("<｜assistant｜>")[-1]
        .replace("<｜end▁of▁conversation｜>", "")
        .replace("<｜human｜>", "")
        .replace("<｜system｜>", "")
        .replace("<｜assistant｜>", "")
        .strip())

    return response

def main():
    """Main function to demonstrate the code generation."""
    print("Setting up the model...")
    model, tokenizer = setup_model()

    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        example_prompt = " ".join(sys.argv[1:])  # Combine all arguments
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