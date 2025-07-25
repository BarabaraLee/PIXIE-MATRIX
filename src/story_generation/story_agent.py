"""
Multi-version Story Generation for PIXIE-MATRIX

Local run under mixie-matrix folder with: python3 -m src.story_generation.story_agent
"""

import json 
from libs.utils import get_gemma_tokenizer_n_model
import openai 
from pathlib import Path 
from typing import List, Dict, Any 
import os 
import torch


class PixieStoryAgent:
    """Generate multiple story versions with chracter awareness"""

    def __init__(self, config_path: str="src/config/book_config.json"):
        self.config = self.load_config(config_path)
        self.characters = self.load_character_registry()
        self.output_dir = Path("src/intermediate_results/story_versions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM
        self.setup_llm(model=self.config["story_config"]["model"])

    def load_config(self, config_path: str) -> dict:
        """Load configuration from a JSON file at the given path."""
        try: 
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            return config_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at path: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in config file: {e}")
        
        
    def load_character_registry(self) -> List[Dict[str, Any]]:
        """Load character registry from a JSON file"""
        character_registry_path = Path("src/intermediate_results/character_registry.json")

        if not character_registry_path.exists():
            print(f"Character registry not found at {character_registry_path}")
            return []

        try:
            with open(character_registry_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            return characters
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing character registry JSON: {e}")
        

    def setup_llm(self, model="Gemma-2b-it"):
        """Setup language model for story generation"""
        print(" Generating story versions")

        if model == "Gemma-2b-it":
            self.tokenizer, self.text_generator = get_gemma_tokenizer_n_model()
            self.model = model 
            self.use_openai = False 
            
        elif model=="gpt-3.5-turbo":
            openai_key = os.getenv("OPEN_API_KEY")
            openai.api_key = openai_key 
            self.model = model 
            self.use_openai = True 

        print(f"Used LLM: {model}")


    def parse_page_response(self, content: str) -> Dict[str, Any]:
        """Parse structured page output from Gemma"""
        lines = content.strip().splitlines()
        result = {"text": "", "image_desc": "", "characters": []}

        for line in lines:
            if line.lower().startswith("story_text:"):
                result["text"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("image_desc:"):
                result["image_desc"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("characters:"):
                chars = line.split(":", 1)[1].strip()
                result["characters"] = [c.strip() for c in chars.split(",") if c.strip()]

        return result


    def generate_page_content(self, prompt: str, page_num: int) -> Dict[str, Any]:
        """Generate story text, image description, and characters using local Gemma"""
        if not self.text_generator or not self.tokenizer:
            raise Exception("Gemma model or tokenizer not loaded.")

        full_prompt = f"""{prompt}

Write page {page_num} of the story.

Format your response *exactly* like this (no extra words):

story_text: <one or two sentences of child-friendly story content>
image_desc: <one sentence describing what to illustrate>
characters: <comma-separated list of character names mentioned on this page>

Make sure:
- All three fields are present
- image_desc is unique and visually descriptive
- characters field includes at least one character (from the list provided)
"""
        character_names = ", ".join([char["name"] for char in self.characters])
        full_prompt += f"\nCharacters available: {character_names}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            self.text_generator.to("cuda")

        with torch.no_grad():
            outputs = self.text_generator.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        if full_prompt.strip() in generated_text:
            generated_text = generated_text.replace(full_prompt.strip(), "").strip()

        return self.parse_page_response(generated_text)


    def generate_story_versions(self) -> List[Dict[str, Any]]:
        """Generate multiple story versions"""
        print(" Generating story versions...")

        versions = []
        num_versions = self.config["story_config"]["versions_to_generate"]

        for i in range(num_versions):
            print(f" Creating story version {i+1}/{num_versions}")

            story_data = self.create_story_version(version_num=i+1)
            versions.append(story_data)

            # Save individual version
            version_file = self.output_dir / f"story_v{i+1}.json"
            with open(version_file, 'w') as f:
                json.dump(story_data, f, indent=2)

        # Save versions summary
        self.save_versions_summary(versions)
        return versions  
    
    
    def create_story_version(self, version_num: int) -> Dict[str, Any]:
        """Create a single story version"""
        # Build character-aware prompt
        prompt = self.build_story_prompt(version_num)

        # Generate story pages
        pages = []
        for page_num in range(1, self.config["story_config"]["pages"] + 1):
            page_content = self.generate_page_content(prompt, page_num)
            pages.append({
                "page_number": page_num,
                "story_text": page_content["text"],
                "image_description": page_content["image_desc"],
                "characters_metioned": page_content["characters"]
            })

        return {
            "version": version_num,
            "title": self.config["title"],
            "subtitle": self.config["subtitle"],
            "author": self.config["author_name"],
            "theme": self.config["story_theme"],
            "pages": pages,
            "characters_used": [char["name"] for char in self.characters]
        }
    
    def build_story_prompt(self, version_num: int) -> str:
        """Build character-aware story generation prompt"""

        character_descriptions = "\n".join([
            f"- {char['name']}: {char['description']} (Personality: {char['personality']})"
            for char in self.characters
        ])

        prompt = f"""
Create a {self.config['story_config']['pages']}-page children's story with the following specifications:

STORY DETAILS:
- Title: {self.config['title']}
- Subtitle: {self.config['subtitle']}
- Theme: {self.config['story_theme']}
- Guidance: {self.config['guidance']}

CHARACTERS TO USE:
{character_descriptions}

REQUIREMENTS:
- Each page should have 1-2 sentences suitable for children
- Include character interactions and development
- Maintain consistency with character personalities
- Version {version_num} should have a unique narrative approach
- Include image descriptions for each page

Please ensure the story is engaging, age-appropriate, and showcases all the characters.
"""
        return prompt 

    def save_versions_summary(self, versions: List[Dict[str, Any]]):
        """Save a summary file containing metadata for all story versions"""
        summary = []

        for version in versions:
            summary.append({
                "version": version["version"],
                "title": version["title"],
                "subtitle": version["subtitle"],
                "theme": version["theme"],
                "characters_used": version["characters_used"],
                "page_count": len(version["pages"])
            })

        summary_path = self.output_dir / "story_versions_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved version summary to: {summary_path}")

if __name__ == "__main__":
    agent = PixieStoryAgent()
    agent.generate_story_versions()