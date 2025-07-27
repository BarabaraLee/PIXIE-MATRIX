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

LoRA weights must be stored locally under the lora/ directory, one folder per character.
Check the hidden temp.py for path to the colab notebook which used GPU to create the trained 
model for Alicaey the kitten and Atley the cub bear.

Environment:
- Requires `HUGGINGFACE_API_TOKEN` in a `.env` file
- Uses config from `src/config/book_config.json` and latest story from `story_collection/`

Usage:
    python3 -m src.image_generation.character_aware_lora_gen
"""
import json
import cv2
from libs.utils import describe_character_appearance, pose_number, position_words, remove_white_background
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.image_generation.cover_gen import generate_cover_image
from libs.constants import NEGATIVE_PROMPT, WIDTH, HEIGHT, NUM_INF_STEPS, GUIDANCE_SCALE
import warnings
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute.*")


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
        # Load local LoRA weights for Alice and Atley from lora/Alicaey and lora/Atley respectively
        pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict="lora/Alicaey" ,
            weight_name="pytorch_lora_weights.safetensors",
            local_files_only=True
            )
        pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict="lora/Atley",
            weight_name="pytorch_lora_weights.safetensors",
            local_files_only=True,
            merge=True
            )
        self.pipeline = pipe

    def canny_edge_map(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges


    def ask_user_to_select_pose(self, char_name: str) -> str:
        folder = self.character_image_root / char_name
        images = sorted(folder.glob(f"{char_name}_pose*.png"), key=pose_number)
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


    def build_story_prompt(self, desc: str, mentioned: List[str]) -> str:

        num_chars = len(mentioned)
        base_scene = (
            "sunny, warm colorful forest, water-color textures."
        )

        if not mentioned:
            character_part = ""
        elif num_chars == 1:
            character_part = f"{mentioned[0]} is standing."
        elif num_chars == 2:
            character_part = f"{mentioned[0]} and {mentioned[1]} are facing each other."
        else:
            names = ", ".join(mentioned[:-1]) + f", and {mentioned[-1]}"
            character_part = f"{names} are together"

        return f"{desc}. {base_scene} {character_part}"


    def generate_images(self):
        import json
        from pathlib import Path

        intermediate_path = Path("src/intermediate_results")
        pose_json_path = intermediate_path / "character_order_pose.json"
        pose_data = {}

        # Try loading existing JSON
        if pose_json_path.exists():
            try:
                with open(pose_json_path, 'r', encoding='utf-8') as f:
                    pose_data = json.load(f)
                    print("Loaded existing character_order_pose.json. Will reuse pose/order data where possible.")
            except Exception as e:
                print(f"⚠️ Failed to read existing character_order_pose.json: {e}")

        # Phase 1: Collect pose selections from user
        for idx, desc in enumerate(self.story["story_descriptions"], 1):
            mentioned = self.get_mentioned_characters(desc)
            if not mentioned:
                continue

            page_key = f"page_{idx:02d}"

            if page_key in pose_data:
                print(f"✅ Skipping input for {page_key}, already present in character_order_pose.json")
                continue

            pose_data[page_key] = {}

            print(f"\n * Page {idx} Story Description:\n{desc}\n")
            print(f"Hint: {mentioned} needs pose image.\n")

            if len(mentioned) > 1:
                print(f"Characters mentioned: {mentioned}")
                print("Please specify display order from left to right (enter comma-separated names):")
                print(f"Example: {', '.join(mentioned)}")
                while True:
                    try:
                        order_input = input("Order: ").strip()
                        order = [name.strip() for name in order_input.split(",")]
                        if sorted(order) == sorted(mentioned):
                            mentioned = order
                            break
                        else:
                            raise ValueError("Names do not match mentioned characters.")
                    except Exception as e:
                        print(f"Invalid input: {e}")

            for char in mentioned:
                pose_path = self.ask_user_to_select_pose(char)
                pose_data[page_key][char] = pose_path

        # Archive old pose JSONs
        i = 1
        while (intermediate_path / f"character_order_pose{i}.json").exists():
            i += 1
        if pose_json_path.exists():
            pose_json_path.rename(intermediate_path / f"character_order_pose{i}.json")
        with open(pose_json_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2)
            print(f"✅ Updated character_order_pose.json with new entries")

        # Phase 2: Generate images
        for idx, (_, desc) in enumerate(zip(self.story["story_sentences"], self.story["story_descriptions"]), 1):
            mentioned = self.get_mentioned_characters(desc)
            page_key = f"page_{idx:02d}"
            if not mentioned or page_key not in pose_data:
                print(f"Skipping page {idx}, no pose data found.")
                continue

            selected_images = []
            for char in mentioned:
                pose_path = pose_data[page_key][char]
                img = cv2.imread(pose_path)
                img_rgba = remove_white_background(img)
                selected_images.append(img_rgba)

            control_image = self.create_control_image(selected_images)
            canny = self.canny_edge_map(control_image[:, :, :3])
            canny_pil = Image.fromarray(canny).convert("RGB")

            layout_hint = ", ".join([f"{name} {pos}" for name, pos in zip(
                mentioned,
                position_words(len(mentioned))
            )]) + ". " if len(mentioned) > 1 else ""

            appearance_hint, negative_background_hint = describe_character_appearance(mentioned)
            prompt = f"{layout_hint}{appearance_hint} {self.build_story_prompt(desc, mentioned)}"

            image = self.pipeline(
                prompt=prompt,
                negative_prompt= negative_background_hint + NEGATIVE_PROMPT,
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
        # Create a transparent RGBA canvas
        canvas = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)

        num_images = len(images)
        target_height = int(HEIGHT * 0.6)  # character height

        positions_x = []
        if num_images == 1:
            positions_x = [WIDTH // 2]
        elif num_images == 2:
            positions_x = [int(WIDTH * 0.25), int(WIDTH * 0.75)]
        else:
            # Spread characters evenly across the width, avoiding edges
            margin = int(WIDTH * 0.1)
            step = (WIDTH - 2 * margin) // (num_images - 1)
            positions_x = [margin + i * step for i in range(num_images)]

        for i, img in enumerate(images):
            # Flip image if needed for inward-facing layout
            midpoint = num_images // 2
            if num_images >= 2:
                if (num_images % 2 == 0 and i < midpoint) or (num_images % 2 == 1 and i < midpoint):
                    img = cv2.flip(img, 1)

            # Resize to target height while preserving aspect ratio
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Compute top-left coordinates (centered horizontally)
            x_center = positions_x[i]
            x_offset = max(0, min(WIDTH - new_w, x_center - new_w // 2))
            y_offset = (HEIGHT - new_h) // 2

            # Ensure paste area is within canvas bounds
            if x_offset + new_w > WIDTH or y_offset + new_h > HEIGHT:
                print(f"⚠️ Skipping character {i} due to size overflow")
                continue

            alpha = resized[:, :, 3] / 255.0
            for c in range(3):
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                    alpha * resized[:, :, c] +
                    (1 - alpha) * canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                )
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3] = (
                np.maximum(canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3], resized[:, :, 3])
            )

        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        return rgb_canvas

    
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
