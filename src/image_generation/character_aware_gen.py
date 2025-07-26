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
        # Load book configuration
        self.config_path = Path("src/config/book_config.json")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Ask for story and character sessions
        self.selected_story = self.load_selected_story()  # from story_collection/
        self.character_session_path = self.select_character_session()  # from character_collection/

        # Load character registry from selected character session
        self.agent = None
        self.ensure_agent_loaded(str(self.character_session_path))
        self.characters = self.load_character_registry(self.character_session_path)

        # Other components  
        self.output_dir = Path("src/intermediate_results/generated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Stable Diffusion + ControlNet pipeline
        self.setup_controlnet()


    def ensure_agent_loaded(self, session_path: str = None):
        if self.agent is None:
            if session_path:
                print(f"[DEBUG] Initializing CharacterGeneratorAgent with session_path={session_path}")
                self.agent = CharacterGeneratorAgent(session_path=session_path)
            else:
                print("[DEBUG] Initializing CharacterGeneratorAgent with NEW session")
                self.agent = CharacterGeneratorAgent()


    def select_character_session(self) -> Path:
        """Ask user to select a character session from character_collection/"""
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
        """Ask user to select a story session and load version_1.json"""
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
        """Heuristically infer characters mentioned in a description."""
        mentioned = []
        description_lower = description.lower()

        for char in self.characters:
            name = char["name"]
            aliases = [name.lower()]
            animal_type = char.get("character_type", "").lower()

            if animal_type:
                aliases.extend([animal_type])

            for alias in aliases:
                if alias in description_lower or difflib.get_close_matches(alias, description_lower.split(), cutoff=0.8):
                    mentioned.append(name)
                    break

        return list(set(mentioned))
    

    def load_character_registry(self, session_path: Path) -> List[Dict[str, Any]]:
        """Load characters required by the book config and ensure session includes all of them."""

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

        # Load backup if any missing characters
        if missing_names:
            print(f"Missing characters: {missing_names}")
            print("Please select backup character images for the missing ones.")
            backup_folder = Path("libs/character_image_lib")
            backup_images = sorted([f for f in backup_folder.glob("*.png")])

            if not backup_images:
                raise FileNotFoundError("No backup images found in libs/character_image_lib.")

            for name in missing_names:
                print(f"\nCharacter: {name}")
                print("Available backup images:")
                for idx, img_path in enumerate(backup_images):
                    print(f"{idx+1}. {img_path.name}")

                while True:
                    try:
                        choice = int(input(f"Choose an image index for {name}: "))
                        selected_img = backup_images[choice - 1]
                        break
                    except Exception as e:
                        print(f"Invalid choice: {e}")

                # Ask for metadata
                self.ensure_agent_loaded(str(session_path))
                metadata = self.agent.suggest_metadata_for_character(name, character_context)

                ctype = metadata.get("character_type") or input(f"Enter character type for {name}: ")
                gender = metadata.get("gender") or input(f"Enter gender for {name}: ")
                appearance = metadata.get("appearance") or input(f"Enter appearance description for {name}: ")
                personality = metadata.get("personality") or input(f"Enter personality for {name}: ")
                notes = metadata.get("additional_notes") or input(f"Additional notes for {name} (optional): ")

                new_path = session_path / f"{name}.png"
                shutil.copyfile(selected_img, new_path)

                new_character = {
                    "name": name,
                    "character_type": ctype,
                    "gender": gender,
                    "appearance": appearance,
                    "personality": personality,
                    "additional_notes": notes,
                    "image_versions": [str(new_path)],
                    "final_image": str(new_path),
                    "created_at": datetime.now().isoformat()
                }

                existing_characters.append(new_character)

        # Normalize all characters' final_image paths and image_versions
        for char in existing_characters:
            img_path = session_path / f"{char['name']}.png"
            char["final_image"] = str(img_path)
            char["image_versions"] = [str(img_path)]
            char["created_at"] = datetime.now().isoformat()

        # Normalize session metadata
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
        """Initialize Stable Diffusion with ControlNet for character consistency"""
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float32
        )

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5-local",  # the local diffusion model
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

    def generate_story_images(self):
        """Generate images for selected story with character consistency"""
        print("Generating character-aware story images...")

        # Generate story page images
        for idx, page_description in enumerate(self.selected_story["page_descriptions"]):
            inferred_characters = self.infer_characters_from_description(page_description)
            if not inferred_characters and ("animal" in page_description or 'people' in page_description):
                inferred_characters = self.characters

            page_data = {
                "page_number": idx + 1,
                "image_description": page_description,
                "characters_mentioned": inferred_characters
            }
            print("page_data", page_data)
            self.generate_page_images(page_data)

        # Generate cover pages
        self.generate_cover_pages()


    def generate_cover_pages(self):
        """Generate main and secondary cover pages based on story descriptions."""
        logger.info("Generating cover pages using descriptions and layout...")

        main_chars = self.config.get("main_characters", [])
        if not main_chars:
            raise ValueError("No main_characters defined in book_config.json.")

        cover1_desc = self.selected_story.get("cover_description_1", "")
        cover2_desc = self.selected_story.get("cover_description_2", "")

        # Use all main characters for both covers
        main_chars = self.config.get("main_characters", [])
        logger.info(f"Using all main characters for cover generation: {main_chars}")
        cover1_chars = main_chars
        cover2_chars = main_chars

        def generate_cover_image(desc, chars, filename):
            character_controls = self.create_character_controls(chars)

            if character_controls is None:
                print(f"Chars: {chars}")
                print(f"filename: {filename}")
                raise ValueError(f"ControlNet image input missing — check character images for: {chars}")

            image = self.pipeline(
                prompt=self.enhance_prompt_with_characters(desc, chars),
                image=Image.fromarray(character_controls).convert("RGB"),
                num_inference_steps=NUM_INF_STEPS,
                guidance_scale=GUIDANCE_SCALE
            ).images[0]
            image_path = self.output_dir / filename
            image.save(image_path)
            logger.info(f"Saved: {image_path}")

        generate_cover_image(cover1_desc, cover1_chars, "cover_1.png")
        generate_cover_image(cover2_desc, cover2_chars, "cover_2.png")

        # Additionally create layout overlay versions
        self.create_cover_layout_pages()

    def create_cover_layout_pages(self):

        title = self.config.get("title", "")
        subtitle = self.config.get("subtitle", "")
        author_name = self.config.get("author_name", "")

        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        try:
            title_font = ImageFont.truetype(font_path, 72)
            subtitle_font = ImageFont.truetype(font_path, 48)
            author_font = ImageFont.truetype(font_path, 32)
        except Exception as e:
            logger.info(f"Error loading fonts: {e}, using default font.")
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            author_font = ImageFont.load_default()

        def get_text_size(text, font):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Main cover layout
        cover_page = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 200, 200))
        draw = ImageDraw.Draw(cover_page)
        title_w, title_h = get_text_size(title, title_font)
        subtitle_w, subtitle_h = get_text_size(subtitle, subtitle_font)
        draw.text(((WIDTH - title_w) / 2, 120), title, fill=(0, 0, 0), font=title_font)
        draw.text(((WIDTH - subtitle_w) / 2, 120 + title_h + 30), subtitle, fill=(0, 0, 0), font=subtitle_font)
        author_text = f"Author: {author_name}"
        author_w, author_h = get_text_size(author_text, author_font)
        draw.text(((WIDTH - author_w) / 2, HEIGHT - author_h - 100), author_text, fill=(0, 0, 0), font=author_font)
        cover_page.save(self.output_dir / "cover_layout_main.png")

        # Secondary layout
        secondary = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 230, 230))
        draw2 = ImageDraw.Draw(secondary)
        block = f"Author: {author_name}\nIllustrated by: {author_name}, assisted by GenAI"
        draw2.multiline_text((80, 150), block, fill=(0, 0, 0), font=subtitle_font, spacing=30)
        secondary.save(self.output_dir / "cover_layout_secondary.png")


    def generate_page_images(self, page_data: Dict[str, Any]):
        """Generate images for a story page with character integration"""
        page_num = page_data["page_number"]
        characters_in_page = page_data["characters_mentioned"]

        # Create character control images
        character_controls = self.create_character_controls(characters_in_page)

        # Generate multiple image options
        for img_idx in range(1, NUM_IMAGES + 1):
            enhanced_prompt = self.enhance_prompt_with_characters(
                page_data["image_description"],
                characters_in_page,
            )

            if character_controls is not None:
                # Use ControlNet with character guidance
                image = self.pipeline(
                    prompt=enhanced_prompt,
                    image=Image.fromarray(character_controls).convert("RGB"),
                    num_inference_steps=NUM_INF_STEPS,
                    guidance_scale=GUIDANCE_SCALE
                ).images[0]

                # Save image
                image_path = self.output_dir / f"page_{page_num:02d}_option_{img_idx}.png"
                image.save(image_path)
                print(f" Generated: {image_path}")

    def create_character_controls(self, character_names: List[Any]) -> np.ndarray:
        """Create control images from character references"""
        if not character_names:
            return None

        # Normalize to list of strings (names)
        normalized_names = []
        for entry in character_names:
            if isinstance(entry, str):
                normalized_names.append(entry)
            elif isinstance(entry, dict) and "name" in entry:
                normalized_names.append(entry["name"])
            else:
                print(f" Unexpected character entry format: {entry}")

        # Load character reference images
        character_images = []
        print("character_names", normalized_names)
        for char_name in normalized_names:
            char_info = next((c for c in self.characters if c["name"] == char_name), None)
            if not char_info:
                print(f" Character info not found for: {char_name}")
                continue

            print("char_info", char_info["final_image"])
            if Path(char_info["final_image"]).exists():
                char_img = cv2.imread(char_info["final_image"])
                character_images.append(char_img)

        if not character_images:
            return None

        # Create composite control image
        composite = self.create_character_composite(character_images)

        # Convert to Canny edge detection for ControlNet
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)

        return canny


    def create_character_composite(self, images: List[np.ndarray]) -> np.ndarray:
        """Resize and stack character images side by side to create a composite"""

        # Determine the common height (e.g., min or max, here we use 512)
        target_height = 512

        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            if h != target_height:
                # Scale width to preserve aspect ratio
                new_width = int((target_height / h) * w)
                resized = cv2.resize(img, (new_width, target_height))
            else:
                resized = img
            resized_images.append(resized)

        return np.hstack(resized_images)

    def enhance_prompt_with_characters(self, base_prompt: str, character_names: List[Any]) -> str:
        """Enhance the prompt with character names for visual guidance"""
        # 🧹 Normalize to names
        normalized_names = []
        for entry in character_names:
            if isinstance(entry, str):
                normalized_names.append(entry)
            elif isinstance(entry, dict) and "name" in entry:
                normalized_names.append(entry["name"])
            else:
                print(f" Unexpected character format in enhance_prompt: {entry}")

        names_str = ", ".join(normalized_names)
        return f"{base_prompt} The scene should clearly feature the characters: {names_str}."


if __name__ == "__main__":
    generator = PixieImageGenerator()
    generator.generate_story_images()
