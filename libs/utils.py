# utils.py

"""Module to place text on images for storybook pages."""
from PIL import Image
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import os
import cv2
import numpy as np
from typing import List, Tuple
import json
from pathlib import Path
import re


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

def cuda_available():
    return torch.cuda.is_available()

def get_torch_dtype():
    return torch.float16 if cuda_available() else torch.float32

def get_device_map():
    return "cuda:0" if cuda_available() else "cpu"

def get_device():
    cuda_avail =  torch.cuda.is_available()
    return "cuda:0" if cuda_avail else "cpu"

# === Functions to load model gemma-2b-it  ===
def get_gemma_tokenizer_n_model():
    
    # === Load Gemma Model ===
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_torch_dtype(),
        device_map=get_device_map(),
        local_files_only=True
    )


    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_torch_dtype(),
        device_map=get_device_map(),
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
    

def describe_character_appearance(names: List[str]) -> Tuple[str, str]:
    """
    Describe up to 2 characters with left/right layout.
    Uses tokenizer to ensure prompt fits within SD-1.5's 77-token limit.
    """
    from pathlib import Path
    import json
    import logging
    from transformers import CLIPTokenizer

    config_path = Path("src/config/book_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    appearance_dict = config.get("character_appearances", {})
    type_dict = config.get("character_types", {
        "Alicaey": "kitten",
        "Atley": "bear cub"
    })

    n = len(names)
    if n == 1:
        hints = [f"{names[0]}, a {appearance_dict.get(names[0], 'gray')} {type_dict.get(names[0], 'animal')} with black eyes."]
    elif n == 2:
        hints = [
            f"{names[0]}: a {appearance_dict.get(names[0], 'gray')} {type_dict.get(names[0], 'animal')} with black eyes.",
            f"{names[1]}: a {appearance_dict.get(names[1], 'gray')} {type_dict.get(names[1], 'animal')} with black eyes."
        ]
    else:
        hints = []

    # Use CLIP tokenizer to trim
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    full_hint = " ".join(hints)
    encoded = tokenizer(full_hint, truncation=True, max_length=77)
    tokens = encoded["input_ids"]
    trimmed_hint = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    all_colors = set(appearance_dict.values())
    all_colors.update([ "gray", "black"])
    color_str = " or ".join(sorted(all_colors))
    negative_background_hint = f"{color_str} leaves, {color_str} sky, {color_str} water, "

    # logging.info(f"appearance_prompt_hint: {trimmed_hint}")
    # logging.info(f"negative_background_hint: {negative_background_hint}")

    # animal_count_hint = f"There should be {n} animal{'s' if n > 1 else ''} in the picture."
    final_hint = f"{trimmed_hint}".strip()
    return final_hint, negative_background_hint


def pose_number(path):
    match = re.search(r"_pose(\d+)(?=\.[a-zA-Z]+$)", path.name)
    return int(match.group(1)) if match else float("inf")