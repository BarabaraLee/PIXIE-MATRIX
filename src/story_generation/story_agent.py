"""
Character-Aware Story Illustration Generator for PIXIE-MATRIX

This module defines the PixieImageGenerator class, which orchestrates the
generation of consistent, character-aware story and cover images using 
Stable Diffusion with ControlNet. It loads character definitions, infers 
which characters appear on each page, applies ControlNet guidance via 
composite edge maps, and renders both story pages and cover images.

Key functionalities:
- Loads story configuration and descriptions from selected story session
- Loads or creates characters with metadata and associated images
- Infers character mentions per page via heuristic text matching
- Generates control images for character consistency
- Produces high-quality illustrations and stylized cover layouts

Designed to support the end-to-end book creation workflow in PIXIE-MATRIX.

Local run under mixie-matrix folder with: python3 -m src.image_generation.character_aware_gen
"""

import json 
import cv2 
from libs.constants import GUIDANCE_SCALE, NUM_IMAGES, NUM_INF_STEPS, WIDTH, HEIGHT
import numpy as np
from pathlib import Path 
from typing import Dict, List, Any 
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.character_generation.character_agent import CharacterGeneratorAgent
from PIL import Image, ImageDraw, ImageFont
import shutil
from datetime import datetime
import difflib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PixieImageGenerator:
    """Generate story images with character consistency"""

    def __init__(self):
        self.config_path = Path("src/config/book_config.json")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.selected_story = self.load_selected_story()
        self.character_session_path = self.select_character_session()
        self.agent = None
        self.ensure_agent_loaded(str(self.character_session_path))
        self.characters = self.load_character_registry(self.character_session_path)

        self.output_dir = Path("src/intermediate_results/generated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.setup_controlnet()

    def ensure_agent_loaded(self, session_path: str = None):
        if self.agent is None:
            self.agent = CharacterGeneratorAgent(session_path=session_path) if session_path else CharacterGeneratorAgent()

    def select_character_session(self) -> Path:
        sessions = sorted([s for s in Path("character_collection").iterdir() if s.is_dir() and s.name.startswith("session_")])
        if not sessions:
            raise FileNotFoundError("No character sessions found.")

        print("Available character sessions:")
        for idx, session in enumerate(sessions):
            print(f"{idx+1}. {session.name}")

        while True:
            try:
                choice = int(input("Select a character session by number: "))
                return sessions[choice - 1]
            except Exception as e:
                print(f"Invalid input: {e}")

    def load_selected_story(self) -> Dict[str, Any]:
        story_dir = Path("story_collection")
        sessions = sorted([s for s in story_dir.iterdir() if s.is_dir() and s.name.startswith("session_")])
        if not sessions:
            raise FileNotFoundError("No story sessions found in story_collection.")

        print("Available story sessions:")
        for idx, session in enumerate(sessions):
            print(f"{idx+1}. {session.name}")

        while True:
            try:
                choice = int(input("Select a story session by number: "))
                selected_path = sessions[choice - 1]
                break
            except Exception as e:
                print(f"Invalid input: {e}")

        story_file = selected_path / "version_1.json"
        if not story_file.exists():
            raise FileNotFoundError(f"version_1.json not found in {selected_path}")

        with open(story_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def infer_characters_from_description(self, description: str) -> List[str]:
        mentioned = []
        description_lower = description.lower()

        for char in self.characters:
            name = char["name"]
            aliases = [name.lower()]
            animal_type = char.get("character_type", "").lower()
            if animal_type:
                aliases.append(animal_type)
            for alias in aliases:
                if alias in description_lower or difflib.get_close_matches(alias, description_lower.split(), cutoff=0.8):
                    mentioned.append(name)
                    break

        return list(set(mentioned))

    def load_character_registry(self, session_path: Path) -> List[Dict[str, Any]]:
        required_character_names = self.config.get("main_characters", [])
        character_context = {
            "theme": self.config.get("story_theme", ""),
            "guidance": self.config.get("guidance", "")
        }

        session_data_path = session_path / "session_data.json"
        if not session_data_path.exists():
            raise FileNotFoundError(f"Missing session_data.json in {session_data_path}")

        with open(session_data_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        existing_characters = session_data.get("characters", [])
        existing_names = {char["name"] for char in existing_characters}
        missing_names = [name for name in required_character_names if name not in existing_names]

        if missing_names:
            print(f"Missing characters: {missing_names}")
            print("Please select backup character images for the missing ones.")
            backup_folder = Path("libs/character_image_lib")
            backup_images = sorted([f for f in backup_folder.glob("*.png")])
            if not backup_images:
                raise FileNotFoundError("No backup images found in libs/character_image_lib.")

            for name in missing_names:
                print(f"\nCharacter: {name}")
                for idx, img_path in enumerate(backup_images):
                    print(f"{idx+1}. {img_path.name}")
                while True:
                    try:
                        choice = int(input(f"Choose an image index for {name}: "))
                        selected_img = backup_images[choice - 1]
                        break
                    except Exception as e:
                        print(f"Invalid choice: {e}")
                self.ensure_agent_loaded(str(session_path))
                metadata = self.agent.suggest_metadata_for_character(name, character_context)

                new_path = session_path / f"{name}.png"
                shutil.copyfile(selected_img, new_path)

                new_character = {
                    "name": name,
                    "character_type": metadata.get("character_type") or input(f"Enter character type for {name}: "),
                    "gender": metadata.get("gender") or input(f"Enter gender for {name}: "),
                    "appearance": metadata.get("appearance") or input(f"Enter appearance for {name}: "),
                    "personality": metadata.get("personality") or input(f"Enter personality for {name}: "),
                    "additional_notes": metadata.get("additional_notes") or input(f"Notes for {name} (optional): "),
                    "image_versions": [str(new_path)],
                    "final_image": str(new_path),
                    "created_at": datetime.now().isoformat()
                }
                existing_characters.append(new_character)

        for char in existing_characters:
            img_path = session_path / f"{char['name']}.png"
            char["final_image"] = str(img_path)
            char["image_versions"] = [str(img_path)]
            char["created_at"] = datetime.now().isoformat()

        session_data = {
            "session_id": session_path.name,
            "created_at": session_path.name.replace("session_", ""),
            "characters": existing_characters
        }

        with open(session_data_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        print("Session data updated and normalized.")
        return existing_characters

    def setup_controlnet(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5-local",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

    def generate_story_images(self):
        print("Generating character-aware story images...")
        for idx, desc in enumerate(self.selected_story["page_descriptions"]):
            inferred = self.infer_characters_from_description(desc)
            if not inferred and ("animal" in desc or "people" in desc):
                inferred = self.characters
            page_data = {"page_number": idx+1, "image_description": desc, "characters_mentioned": inferred}
            self.generate_page_images(page_data)
        self.generate_cover_pages()

    def generate_cover_pages(self):
        logger.info("Generating cover pages using descriptions and layout...")
        cover1_desc = self.selected_story.get("cover_description_1", "")
        cover2_desc = self.selected_story.get("cover_description_2", "")
        main_chars = self.config.get("main_characters", [])
        for desc, chars, name in [(cover1_desc, main_chars, "cover_1.png"), (cover2_desc, main_chars, "cover_2.png")]:
            control = self.create_character_controls(chars)
            if control is None:
                raise ValueError(f"Missing ControlNet input for characters: {chars}")
            control_pil = Image.fromarray(control).convert("RGB")
            image = self.pipeline(prompt=self.enhance_prompt_with_characters(desc, chars), image=control_pil, num_inference_steps=NUM_INF_STEPS, guidance_scale=GUIDANCE_SCALE).images[0]
            image.save(self.output_dir / name)
        self.create_cover_layout_pages()

    def create_cover_layout_pages(self):
        title = self.config.get("title", "")
        subtitle = self.config.get("subtitle", "")
        author = self.config.get("author_name", "")
        try:
            font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
            title_font = ImageFont.truetype(font_path, 72)
            subtitle_font = ImageFont.truetype(font_path, 48)
            author_font = ImageFont.truetype(font_path, 32)
        except Exception:
            title_font = subtitle_font = author_font = ImageFont.load_default()

        def draw_text(image, title, subtitle, author):
            draw = ImageDraw.Draw(image)
            title_w, title_h = title_font.getsize(title)
            subtitle_w, subtitle_h = subtitle_font.getsize(subtitle)
            draw.text(((WIDTH - title_w)/2, 120), title, fill=(0,0,0), font=title_font)
            draw.text(((WIDTH - subtitle_w)/2, 120 + title_h + 30), subtitle, fill=(0,0,0), font=subtitle_font)
            auth_text = f"Author: {author}"
            aw, ah = author_font.getsize(auth_text)
            draw.text(((WIDTH - aw)/2, HEIGHT - ah - 100), auth_text, fill=(0,0,0), font=author_font)

        cover_main = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 200, 200))
        draw_text(cover_main, title, subtitle, author)
        cover_main.save(self.output_dir / "cover_layout_main.png")

        block_text = f"Author: {author}\nIllustrated by: {author}, assisted by GenAI"
        secondary = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 230, 230))
        draw = ImageDraw.Draw(secondary)
        draw.multiline_text((80, 150), block_text, fill=(0,0,0), font=subtitle_font, spacing=30)
        secondary.save(self.output_dir / "cover_layout_secondary.png")

    def generate_page_images(self, page_data: Dict[str, Any]):
        page_num = page_data["page_number"]
        characters = page_data["characters_mentioned"]
        control = self.create_character_controls(characters)
        if control is not None:
            control_pil = Image.fromarray(control).convert("RGB")
            for idx in range(1, NUM_IMAGES + 1):
                prompt = self.enhance_prompt_with_characters(page_data["image_description"], characters)
                image = self.pipeline(prompt=prompt, image=control_pil, num_inference_steps=NUM_INF_STEPS, guidance_scale=GUIDANCE_SCALE).images[0]
                image.save(self.output_dir / f"page_{page_num:02d}_option_{idx}.png")

    def create_character_controls(self, character_names: List[Any]) -> np.ndarray:
        if not character_names:
            return None
        normalized = [entry if isinstance(entry, str) else entry.get("name", "") for entry in character_names]
        images = []
        for name in normalized:
            char = next((c for c in self.characters if c["name"] == name), None)
            if char and Path(char["final_image"]).exists():
                img = cv2.imread(char["final_image"])
                if img is not None:
                    images.append(img)
        if not images:
            return None
        return self.create_character_composite(images)

    def create_character_composite(self, images: List[np.ndarray]) -> np.ndarray:
        target_height = 512
        resized = []
        for img in images:
            h, w = img.shape[:2]
            new_w = int((target_height / h) * w) if h != target_height else w
            resized.append(cv2.resize(img, (new_w, target_height)))
        return np.hstack(resized)

    def enhance_prompt_with_characters(self, base: str, names: List[Any]) -> str:
        normalized = [entry if isinstance(entry, str) else entry.get("name", "") for entry in names]
        return f"{base} The scene should clearly feature the characters: {', '.join(normalized)}."


if __name__ == "__main__":
    generator = PixieImageGenerator()
    generator.generate_story_images()
