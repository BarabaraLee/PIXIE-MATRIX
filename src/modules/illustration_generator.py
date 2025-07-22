# filename: illustration_generator.py
"""Module to generate illustrations for story sentences using DALL·E or Hugging Face's Stable Diffusion XL + ControlNet (if sketch is provided)."""

import openai
import os
from dotenv import load_dotenv
from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from transformers.utils import logging
logging.set_verbosity_info()

# Load environment variables from .env file
load_dotenv()

# constants
HEIGHT = 768 # 1536 
WIDTH = 512 # 1024
NUM_INF_STEPS = 20
NUM_IMAGES = 2
NEGATIVE_PROMPT = ("blurry, distorted, messy, extra limbs, missed limbs,"
                   "misplaced limbs, wrong number of fingers, horror, dark," 
                   "low quality, surreal, mammals in sky， reptiles in sky,"
                   "bad anatomy, bad proportions, bad hands, bad fingers, bad face,"
                   "trees in sky, plants in sky, flowers in sky, buildings in sky"
                   )
GUIDANCE_SCALE = 8.5
IMAGE_STYLE = "Watercolor"

def generate_negative_prompt():
    """Generate a comprehensive negative prompt to avoid common pitfalls in image generation."""
    negative_prompt_categories = {
        "Image Quality": "blurry, low quality, grainy, noisy",
        "Anatomy & Faces": "bad anatomy, extra fingers, mutated hands, cloned face, missing fingers, missing arms, broken limbs, malformed, twisted, mutated face, extra limbs, multiple heads, ugly eyes, extra eyes",
        "Layout": "cropped, out of frame, bad proportions, bad perspective",
        "Style Control": "unrealistic, photorealistic, photo, photographic, 3d, cgi, shiny skin",
        "Theme Filtering": "horror, creepy, human sexy, human nudity, gory, nsfw, adult, violent",
        "Animal Transformation": "animal with clothes, animal with human face, animal with human body, animal with human hands, animal with human legs, animal with human arms",
        "Visual Artifacts": "text, watermark, signature, glitch",
        "Mood & Tone": "dark, harsh lighting, dark shadows, b&w, monochrome",
        "Composition": "cluttered, messy, chaotic, disorganized, busy, overcrowded",
        "Lighting": "overexposed, underexposed, harsh shadows, blown highlights, poor lighting, uneven lighting, dark shadows",
        "Object Counts": "wrong number of mentioned objects, wrong number of mentioned people, wrong number of mentioned animals",
        "Animal Features": "animal replaced by human, animal with human features, animal with human face, animal with human body, animal with human hands, animal with human legs, animal with human arms",
    }
    negative_prompt = ", ".join(negative_prompt_categories.values())
    return negative_prompt


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
                size=f"{WIDTH}x{HEIGHT}"
            )
            image_url = response['data'][0]['url']
            illustrations.append(image_url)

    elif model == "huggingface-SDXL":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if sketch_map:
            print("Using Stable Diffusion XL with ControlNet for sketch guidance.")
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

        else:
            print("Using Stable Diffusion XL without ControlNet.")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32)

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
            prompt = f"An illustration for a toddler book: {sentence}. {IMAGE_STYLE} style. Colorful, soft edges, warm pastel tones, kid-friendly drawing style, portrait layout."
            print(f"Generating illustration for page {i + 1}: {prompt}")
            control_image = None
            if sketch_map:
                for sketch_label, sketch_data in sketch_map.items():
                    if (i + 1) in sketch_data["pages"]:
                        control_image = sketch_control_images[sketch_label]
                        break
            
            negative_prompt = generate_negative_prompt()
            if control_image is not None:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
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
                    negative_prompt=negative_prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INF_STEPS,
                    num_images_per_prompt=NUM_IMAGES,
                    guidance_scale=GUIDANCE_SCALE,
                )

            if len(result.images) == 1:
                img = result.images[0]
                filename = f"./page_images/page_{i+1:02d}_sdxl.png"
                img.save(filename)
                illustrations.append(filename)
            elif len(result.images) > 1:
                filenames = []
                for j, img in enumerate(result.images):
                    filename = f"./page_images/page_{i+1:02d}_sdxl_{j+1}.png"
                    img.save(filename)
                    filenames.append(filename)
                illustrations.append(filenames)

    else:
        raise ValueError(f"Unsupported model: {model}")

    return illustrations
