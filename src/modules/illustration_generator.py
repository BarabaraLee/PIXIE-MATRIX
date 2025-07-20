# filename: illustration_generator.py
"""Module to generate illustrations for story sentences using DALL·E or Hugging Face's Stable Diffusion XL + ControlNet (if sketch is provided)."""

import openai
import os
from dotenv import load_dotenv
from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers.utils import logging
logging.set_verbosity_info()

# Load environment variables from .env file
load_dotenv()

def preprocess_sketch_for_controlnet(sketch_path):
    image = cv2.imread(sketch_path)
    image = cv2.Canny(image, 100, 200)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image)

def generate_illustrations(story_sentences, model="huggingface-SDXL", sketch_map=None):
    """Generate illustrations for each sentence using DALL·E or SDXL + ControlNet if sketch is provided.

    Args:
        story_sentences (list): List of story sentences to illustrate.
        model (str): The model to use for illustration.
        sketch_map (dict or None): Dictionary like:
            {
            "sketch1": {"path": "path/to/sketch1.png", "pages": [1, 2, 3]},
            "sketch2": {"path": "path/to/sketch2.png", "pages": [4, 5, 6]},
            ...
            }

    Returns:
        list: List of generated illustration file paths.
    """

    if model == "dall-e-3":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = api_key

        illustrations = []
        for i, sentence in enumerate(story_sentences):
            prompt = f"An illustration for a toddler book: {sentence}. Make it colorful, engaging, and suitable for children."
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1536"
            )
            image_url = response['data'][0]['url']
            illustrations.append(image_url)

    elif model == "huggingface-SDXL":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        illustrations = []

        # Preprocess control images once for reuse
        sketch_control_images = {}
        if sketch_map:
            for sketch_label, sketch_data in sketch_map.items():
                image_path = sketch_data["path"]
                control_image = preprocess_sketch_for_controlnet(image_path)
                sketch_control_images[sketch_label] = control_image

        for i, sentence in enumerate(story_sentences):
            prompt = f"An illustration for a toddler book: {sentence}. Colorful, soft edges, warm pastel tones, kid-friendly drawing style, portrait layout."

            control_image = None
            if sketch_map:
                for sketch_label, sketch_data in sketch_map.items():
                    if (i + 1) in sketch_data["pages"]:
                        control_image = sketch_control_images[sketch_label]
                        break

            if control_image is not None:
                result = pipe(
                    prompt=prompt,
                    image=control_image,
                    height=1536,
                    width=1024,
                    num_inference_steps=30
                )
            else:
                result = pipe(
                    prompt=prompt,
                    height=1536,
                    width=1024,
                    num_inference_steps=30
                )

            img = result.images[0]
            filename = f"page_{i+1:02d}_sdxl.png"
            img.save(filename)
            illustrations.append(filename)

    else:
        raise ValueError(f"Unsupported model: {model}")

    return illustrations
