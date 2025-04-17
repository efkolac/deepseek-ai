import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import Optional
from pathlib import Path
import logging
import shutil
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
# MODEL_DIR = Path("models/DeepSeek-R1-Distill-Llama-8B")
# MODEL_DIR = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
MODEL_DIR = Path(os.path.join(os.environ.get("MODEL_BASE_PATH", "models"), "DeepSeek-R1-Distill-Qwen-1.5B"))
LOCAL_MODEL_DIR = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")  # Custom local folder

tokenizer = None
model = None
def download_model():
    """Force-download the model to a local directory (no cache)"""
    try:
        # Create fresh directory
        if LOCAL_MODEL_DIR.exists():
            shutil.rmtree(LOCAL_MODEL_DIR)  # Delete existing
        LOCAL_MODEL_DIR.mkdir(parents=True)

        logger.info(f"Downloading {MODEL_DIR} to {LOCAL_MODEL_DIR}...")

        # Download all required files
        required_files = [
            "config.json",
            "model.safetensors",  # or "pytorch_model.bin"
            "tokenizer.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]

        for file in required_files:
            hf_hub_download(
                repo_id=MODEL_DIR,
                filename=file,
                local_dir=LOCAL_MODEL_DIR,
                force_download=True,  # Ignore cache
                resume_download=False  # Fresh download
            )

        logger.info("Download complete!")
        return LOCAL_MODEL_DIR

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def load_model():
    """ Load model and tokenizer """
    global tokenizer, model
    logger.info("Loading model...")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model 4 directory not found at {MODEL_DIR}")
    
    try:
        # Step 1: Force download (no cache)
        model_path = download_model()

        # Step 2: Load from local files
        logger.info("Loading model from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info(f"Model loaded on {model.device}")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Load model when the container starts
load_model()

def handler(event):
    """
    RunPod serverless handler function
    Expected input format:
    {
        "input": {
            "prompt": "Your question here",
            "context": "Optional context",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    """
    try:
        input_data = event.get('input', {})
        
        # Validate input
        if 'prompt' not in input_data:
            return {"error": "Missing required field 'prompt' in input"}
        
        # Get generation parameters with defaults
        prompt = input_data['prompt']
        context = input_data.get('context')
        max_length = input_data.get('max_length', 2048)
        temperature = input_data.get('temperature', 0.7)
        top_p = input_data.get('top_p', 0.9)
        
        folder_path = input_data.get('folder_path', './context_files')  # Default folder
        
        # Read and append files from folder if it exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            file_contents = []
            for file_name in sorted(os.listdir(folder_path)):
                file_path = Path(folder_path) / file_name
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_contents.append(f"File: {file_name}\n{f.read().strip()}")
                    except Exception as e:
                        logger.warning(f"Could not read file {file_name}: {str(e)}")
            
            if file_contents:
                files_context = "\n\n".join(file_contents)
                context = f"{context}\n\n{files_context}" if context else files_context
        
        # Format prompt
        if context:
            formatted_prompt = f"<s>[INST] <context>\n{context}\n</context>\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        
        return {"response": response}
    
    except torch.cuda.OutOfMemoryError:
        return {"error": "GPU out of memory - try reducing max_length"}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return {"error": str(e)}

if __name__ == '__main__':
    # test_event = {
    #     "input": {
    #         "prompt": "Explain quantum computing in simple terms",
    #         "max_length": 200,
    #         "temperature": 0.7
    #     }
    # }
    
    # # Simulate RunPod's call
    # result = handler(test_event)
    # print("Test Output:", result)
    runpod.serverless.start({
        "handler": handler
    })