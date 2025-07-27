# utils.py

"""Module to place text on images for storybook pages."""
from PIL import Image
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import os
import cv2
import numpy as np
from typing import List
import json
from pathlib import Path


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


def remove_white_background(img_bgr, white_thresh=245):
    """
    Converts white areas in the image to transparent alpha (RGBA format).
    """
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    lower = np.array([white_thresh, white_thresh, white_thresh, 0], dtype=np.uint8)
    upper = np.array([255, 255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(img_rgba, lower, upper)
    img_rgba[white_mask == 255, 3] = 0
    return img_rgba


def position_words(n: int) -> List[str]:
    if n == 2:
        return ["left", "right"]
    elif n == 3:
        return ["left", "center", "right"]
    else:
        # For >3 characters, just return "left", "middle1", "middle2", ..., "right"
        words = ["left"]
        words += [f"middle{i}" for i in range(1, n-1)]
        words.append("right")
        return words
    

def describe_character_appearance(names: List[str]) -> str:
    """
    In book_config.json, we have info as follows:
        {
        "main_characters": ["Atley", "Aliceay"],
        "character_appearances": {
            "Atley": "brown color",
            "Aliceay": "yellow color"
        },
        ...
        }
    This function reads info under "character_appearances" which 
    encode the skin/fur color of the animal.
    """
    config_path = Path("src/config/book_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    appearance_dict = config.get("character_appearances", {})
    appearance_hint = " ".join([
        f"{name} is {appearance_dict.get(name, 'a character')}."
        for name in names
    ])

    color_str = " or ".join( 
        [x.replace("color", "") for x in appearance_dict.values()] 
        )
    # this is to reduce hallucination on background color
    negative_background_hint =  f"{color_str} background color, "

    return appearance_hint, negative_background_hint