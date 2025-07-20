# filename: illustration_generator.py
"""Module to generate illustrations for story sentences using DALL·E or Hugging Face's Stable Diffusion XL."""

import openai
# import replicate  # Uncomment if using Replicate in the future
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def generate_illustrations(story_sentences, model="huggingface-SDXL"):
    """Generate illustrations for each sentence in the story using DALL·E or SDXL.

    Args:
        story_sentences (list): List of story sentences to illustrate.
        model (str): The model to use for illustration ("dall-e-3" or "huggingface-SDXL"),
            "huggingface-SDXL" being the default.

    Returns:
        list: List of generated illustration URLs or file paths.
    """

    if model == "dall-e-3":
        # Ensure the API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = api_key

        # start of image generation
        illustrations = []
        for i, sentence in enumerate(story_sentences):
            prompt = f"An illustration for a toddler book: {sentence}. Make it colorful, engaging, and suitable for children."
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1536"  # Portrait orientation for book pages
            )
            image_url = response['data'][0]['url']
            illustrations.append(image_url)

    elif model == "huggingface-SDXL":
       # Ensure the API key is set
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN is not set in the environment variables.")

        # start of image generation
        illustrations = []
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"}

        for i, sentence in enumerate(story_sentences):
            prompt = f"An illustration for a toddler book: {sentence}. Colorful, soft edges, warm pastel tones, kid-friendly drawing style, portrait layout."

            payload = {
                "inputs": prompt,
                "parameters": {
                    "height": 1536,
                    "width": 1024,
                    "num_inference_steps": 30  # optional
                }
            }

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                filename = f"page_{i+1:02d}_hf.png"
                with open(filename, "wb") as f:
                    f.write(response.content)
                illustrations.append(filename)
            else:
                print(f"Error generating image for page {i+1}: {response.status_code}")
                illustrations.append(None)

    else:
        raise ValueError(f"Unsupported model: {model}")
    
    return illustrations
