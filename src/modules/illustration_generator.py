# filename: illustration_generator.py
"""Module to generate illustrations for story sentences using DALL·E or Hugging Face's Stable Diffusion XL + ControlNet (if sketch is provided)."""

import openai
import os
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from libs.constants import HEIGHT, WIDTH, NUM_INF_STEPS, NUM_IMAGES, NEGATIVE_PROMPT, GUIDANCE_SCALE, IMAGE_STYLE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def preprocess_sketch_for_controlnet(sketch_path):
    image = cv2.imread(sketch_path)
    image = cv2.Canny(image, 100, 200)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image)

def generate_prompt(sentence, style=IMAGE_STYLE, is_cover=False, title=""):
    """Generate a prompt for the illustration model based on the sentence. 
    Token info is only logged, not used"""
    cover_addon = " for a cover page" if is_cover else ""
    cover_addon += f" including the main character - {title}"
    prompt = f"A colorful illustration for a toddler book{cover_addon} : {sentence} {style}"

    return prompt

def generate_illustrations(story_sentences, model="huggingface-SDXL", sketch_map=None, title=""):
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
            prompt = generate_prompt(sentence, IMAGE_STYLE, (i < 2), title)
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=f"{WIDTH}x{HEIGHT}"
            )
            image_url = response['data'][0]['url']
            illustrations.append(image_url)

    elif model == "huggingface-SDXL":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if sketch_map:
            logger.info("Using Stable Diffusion XL with ControlNet for sketch guidance.")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float32
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float32
            ).to(device)

        else:
            logger.info("Using Stable Diffusion XL without ControlNet.")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                torch_dtype=torch.float32
            ).to(device)
            print(pipe.device)
            print(next(pipe.unet.parameters()).dtype)

        pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload() # use this only if you have GPU

        illustrations = []

        # Preprocess control images once for reuse
        sketch_control_images = {}
        if sketch_map:
            for sketch_label, sketch_data in sketch_map.items():
                image_path = sketch_data["path"]
                control_image = preprocess_sketch_for_controlnet(image_path)
                sketch_control_images[sketch_label] = control_image

        for i, sentence in enumerate(story_sentences):
            prompt = generate_prompt(sentence, IMAGE_STYLE, (i < 2), title)
            logger.info(f"Generating illustration for page {i + 1}: {prompt}")
            control_image = None
            if sketch_map:
                for sketch_label, sketch_data in sketch_map.items():
                    if (i + 1) in sketch_data["pages"]:
                        control_image = sketch_control_images[sketch_label]
                        break
            
            if control_image is not None:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=control_image,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INF_STEPS,
                    num_images_per_prompt=NUM_IMAGES,
                    guidance_scale=GUIDANCE_SCALE,
                )
            else:
                result = pipe( 
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INF_STEPS,
                    num_images_per_prompt=NUM_IMAGES,
                    guidance_scale=GUIDANCE_SCALE,
                )

            if len(result.images) == 1:
                img = result.images[0]
                filename = f"./page_images/page_{i+1:02d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_sdxl.png"
                img.save(filename)
                illustrations.append(filename)
            elif len(result.images) > 1:
                filenames = []
                for j, img in enumerate(result.images):
                    filename = f"./page_images/page_{i+1:02d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_sdxl_{j+1}.png"
                    img.save(filename)
                    filenames.append(filename)
                illustrations.append(filenames)

    else:
        raise ValueError(f"Unsupported model: {model}")

    return illustrations
