# sketch_to_prompt_generator.py

"""
This script ingests one or more sketch/skeleton images (e.g., pencil layout of a room or landscape) and generates a dictionary of labeled text prompts describing each image.
The output can be used to guide Stable Diffusion or other generative models.

The process involves:
1. Image preprocessing
2. Captioning via BLIP
3. Contextual prompt refinement
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import argparse

# Load BLIP model for visual captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
model.to("cpu")  # Ensure CPU-only mode

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cpu")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def refine_caption_with_context(raw_caption, sketch_type="room layout"):
    return (
        f"A sketch showing {raw_caption}. It appears to depict a {sketch_type}, "
        f"with objects roughly positioned. Please render a realistic image based on this layout."
    )

def sketch_to_prompt_dict(labeled_sketches):
    """
    Args:
        labeled_sketches (list of tuples): [(label, image_path, sketch_type), ...]
    Returns:
        dict: {label: refined_prompt}
    """
    prompts = {}
    for label, image_path, sketch_type in labeled_sketches:
        raw_caption = generate_caption(image_path)
        refined_prompt = refine_caption_with_context(raw_caption, sketch_type)
        prompts[label] = refined_prompt
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled prompts from sketch images")
    parser.add_argument("--sketches", nargs="+", help="List of label::path::type strings")
    args = parser.parse_args()

    labeled_inputs = []
    for entry in args.sketches:
        try:
            label, path, sketch_type = entry.split("::")
            labeled_inputs.append((label.strip(), path.strip(), sketch_type.strip()))
        except ValueError:
            print(f"Invalid input format: {entry}. Expected 'label::path::sketch_type'. Skipping.")

    prompt_dict = sketch_to_prompt_dict(labeled_inputs)
    for label, prompt in prompt_dict.items():
        print(f"\n[{label}]\n{prompt}")
