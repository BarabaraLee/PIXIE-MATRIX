# utils.py

"""Module to place text on images for storybook pages."""
from PIL import Image
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_image_type(img):
    """Check if the input is a PIL Image or a file path."""
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        try:
            return Image.open(img)
        except Exception as e:
            logger.error(f"Error opening image file {img}: {e}")
            return None

# === Functions to load model gemma-2b-it  ===
def get_gemma_tokenizer_n_model():

    cuda_available = torch.cuda.is_available()
    
    # === Load Gemma Model ===
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )


    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if cuda_available else torch.float32,
        device_map="auto" if cuda_available else "cpu",
        local_files_only=True
    )
    return tokenizer, model_gemma