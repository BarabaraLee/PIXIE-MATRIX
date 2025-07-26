"""
Character-Aware Storybook Image Generator using LoRA + ControlNet

This script defines the CharacterAwareLoRAGenerator class, which generates
storybook illustrations and cover images with character consistency. It uses:

- Stable Diffusion with ControlNet for structure guidance via Canny edges
- LoRA fine-tuned weights from Hugging Face for character-specific appearance

Main Features:
- Prompts the user to select character pose images for each story page
- Constructs composite control images from selected character poses
- Applies ControlNet + LoRA pipeline to generate illustrations
- Generates front and back cover images using the same character setup

Assumes character pose images are stored under:
  libs/character_image_lib/{character_name}/{character_name}_pose{x}.png

LoRA weights must be available on Hugging Face Hub and referenced in `setup_pipeline`.

Environment:
- Requires `HUGGINGFACE_API_TOKEN` in a `.env` file
- Uses config from `src/config/book_config.json` and latest story from `story_collection/`

Usage:
    python3 -m src.image_generation.character_aware_lora_gen
"""
import os
import json
import cv2
import torch
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from huggingface_hub import login
from src.image_generation.cover_gen import generate_cover_image
from libs.constants import CHARACTOR_HEIGHT, WIDTH, HEIGHT, NUM_INF_STEPS, GUIDANCE_SCALE

# Load .env for token
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(token=hf_token)


class CharacterAwareLoRAGenerator:
    def __init__(self, config_path="src/config/book_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.story = self.load_story()
        self.characters = self.config["main_characters"]
        self.character_image_root = Path("libs/character_image_lib")
        self.output_dir = Path("src/intermediate_results/generated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_pipeline()

    def load_story(self) -> Dict[str, Any]:
        session_folder = sorted(Path("story_collection").iterdir())[-1]
        with open(session_folder / "version_1.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def setup_pipeline(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to("cpu")

        # Inject LoRA
        lora_path = "lora"  # Replace with your real LoRA repo
        pipe.load_lora_weights(lora_path)
        self.pipeline = pipe

    def canny_edge_map(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def ask_user_to_select_pose(self, char_name: str) -> str:
        folder = self.character_image_root / char_name
        images = sorted(folder.glob(f"{char_name}_pose*.png"))
        if not images:
            raise FileNotFoundError(f"No poses found for {char_name} in {folder}")
        print(f"Select pose for character '{char_name}':")
        for idx, img in enumerate(images):
            print(f"{idx+1}. {img.name}")
        while True:
            try:
                choice = int(input("Enter pose number: "))
                return str(images[choice - 1])
            except Exception as e:
                print(f"Invalid input: {e}")

    def generate_images(self):
        for idx, (sentence, desc) in enumerate(zip(self.story["story_sentences"], self.story["page_descriptions"]), 1):
            mentioned = self.get_mentioned_characters(desc)
            if not mentioned:
                print(f"Skipping page {idx}, no main characters mentioned.")
                continue

            selected_images = []
            for char in mentioned:
                pose_path = self.ask_user_to_select_pose(char)
                img = cv2.imread(pose_path)
                selected_images.append(img)

            control_image = self.create_control_image(selected_images)
            canny = self.canny_edge_map(control_image)
            canny_pil = Image.fromarray(canny).convert("RGB")

            prompt = f"{desc} Show characters: {', '.join(mentioned)}."
            image = self.pipeline(
                prompt=prompt,
                image=canny_pil,
                num_inference_steps=NUM_INF_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH
            ).images[0]

            out_path = self.output_dir / f"page_{idx:02d}.png"
            image.save(out_path)
            print(f" Saved {out_path}")

    def get_mentioned_characters(self, desc: str) -> List[str]:
        desc = desc.lower()
        return [name for name in self.characters if name.lower() in desc]

    def create_control_image(self, images: List[np.ndarray]) -> np.ndarray:
        target_height = CHARACTOR_HEIGHT
        resized = []
        for img in images:
            h, w = img.shape[:2]
            scale = target_height / h
            resized_img = cv2.resize(img, (int(w * scale), target_height))
            resized.append(resized_img)
        return np.hstack(resized)
    
    def generate_covers(self):
        cover1 = self.story["cover_description_1"]
        cover2 = self.story["cover_description_2"]

        generate_cover_image(
            prompt=cover1,
            character_names=self.characters,
            pipeline=self.pipeline,
            character_image_root=self.character_image_root,
            output_path=self.output_dir / "cover_1.png"
        )
        generate_cover_image(
            prompt=cover2,
            character_names=self.characters,
            pipeline=self.pipeline,
            character_image_root=self.character_image_root,
            output_path=self.output_dir / "cover_2.png"
        )

if __name__ == "__main__":
    generator = CharacterAwareLoRAGenerator()
    generator.generate_images()
    generator.generate_covers()
