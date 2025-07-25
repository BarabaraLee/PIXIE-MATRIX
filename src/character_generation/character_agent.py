"""
Character Generation Agent System

An intelligent agent system that helps users create story characters through
an interactive process involving character specification, image generation,
feedback collection, and iterative refinement.

Local run under mixie-matrix folder with: python3 -m src.character_generation.character_agent

"""

import os 
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional 
from dataclasses import dataclass, asdict 
import logging
from libs.constants import NEGATIVE_PROMPT, NUM_INF_STEPS, GUIDANCE_SCALE

from libs.utils import get_gemma_tokenizer_n_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try: 
    from diffusers import StableDiffusionPipeline
    import torch 
    LOCAL_MODELS_AVAILABLE = True 
except ImportError:
    LOCAL_MODELS_AVAILABLE = False 

@dataclass 
class Character:
    """Data class to represent a character"""
    name: str 
    character_type: str 
    gender: str 
    appearance: str 
    personality: str 
    additional_notes: str 
    image_versions: List[str]
    final_image: Optional[str] = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat() 


class CharacterGeneratorAgent:
    """Main agent class for character generation with local model support"""

    def __init__(self):
        """Initialize the character generator agent"""
        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.characters: List[Character] = []
        self.character_folder = "character_collection"
        self.session_folder = f"session_{self.now}"
        self.working_folder = os.path.join(self.character_folder, self.session_folder)

        # Create directories 
        os.makedirs(self.working_folder, exist_ok=True)

        # Initialize image generation method
        self.image_generator = None 
        self.generator_type = None 
        self.generator_type = "demo"

        # Try setup local models
        try: 
            self.setup_local_models()
            self.generator_type = "local_models"
            print("Local models (SD + Gemma) configured")
        except Exception as e:
            print(f"Failed to load local models: {e}")
            raise Exception("Local models failed to initialize")
        
    def setup_local_models(self):
        """Setup local Stable Diffusion and Gemma models"""
        print("Loading local models")

        self.image_generator = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            touch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        if torch.cuda.is_available():
            self.image_generator = self.image_generator.to("cuda")

        print("Loaded Stable Diffusion from Local.")

        self.gemma_tokenizer , self.text_generator = get_gemma_tokenizer_n_model() 

        print("Loaded Local Gemma-2B-IT from Local.")

    def start_character_creation_session(self):
        """Start the main character creation workflow"""
        print("Welcome to the Character Generation Agent System!")
        print("=" * 60)
        print("I'll help you create characters for your story book.\n")

        # Ask for number of characters
        num_characters = self.ask_for_character_count()

        # Create each character
        for i in range(num_characters):
            print(f"Creating Character {i + 1} of {num_characters}")
            print("-" * 40)
            character = self.create_character(i + 1)
            self.characters.append(character)

        # Final summary
        self.show_final_summary()
        self.save_session_data()

    def ask_for_character_count(self) -> int:
        """Ask user how many characters they want to create"""
        while True:
            try:
                count = input("How many characters would you like to create for your story? ")
                num = int(count)
                if num > 0:
                    print(f"Great! We'll create {num} character{'s' if num > 1 else ''}.")
                    return num 
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

    def create_character(self, character_number: int) -> Character:
        """Create a single character through interactive process"""
        print(f"Let's create character #{character_number}")

        # Collect basic information
        character_info = self.collect_character_info()

        # Create chracter object
        character = Character(
            name=character_info['name'],    
            character_type=character_info['type'],
            gender=character_info['gender'],
            appearance=character_info['appearance'],
            personality=character_info['personality'],
            additional_notes=character_info['notes'],
            image_versions=[]
        )

        # Generate and refine images
        self.generate_and_refine_character_image(character)

        return character
    
    def collect_character_info(self) -> Dict[str, str]:
        """Collect character information from user"""
        info = {}

        # Character name
        info['name'] = input("What's the character's name? ").strip()

        # Character type
        print("\n What type of character is this?")
        print(" Examples: human, cat, dog, dragon, fairy, robot, etc.")
        info["type"] = input("Type: ").strip()

        # Gender
        print("\n What's the character's gender?")
        print(" Examples: male, female, animal.")
        info["gender"] = input("Gender: ").strip()

        # Appearance
        print(f"\n Describe {info["name"]}'s physical appearance:")
        print(" Include details like: hair color, eye color, clothing style, size, etc.")
        info["appearance"] = input("Appearance: ").strip()

        # Personality
        print(f"\n Describe {info['name']}'s personality:")
        print(" Examples: friendly, shy, brave, mischievous, wise, etc.")
        info['personality'] = input("Personality: ").strip()

        # Additional notes
        print(f"\n Any additional notes about {info['name']}?")
        print(" Special abilities, backstory, role in story, etc. (optional)")
        info["notes"] = input("Additional notes: ").strip()

        return info 
    
    def generate_and_refine_character_image(self, character: Character):
        """Generate character image and refine based on user feedback"""
        print(f"\n Generating image for {character.name}...")

        version = 1
        while True:
            # Generate image
            image_path = self.generate_character_image(character, version)
            character.image_versions.append(image_path)

            print(f"Generated image version {version} for {character.name}")
            print(f"Saved as: {image_path}")

            # Ask for feedback
            satisfied = self.get_user_feedback(character, version)

            if satisfied: 
                character.final_image = image_path 
                print(f" Character {character.name} is finalized!")
                break 
            else:
                # Ask for modifications
                modifications = self.ask_for_modifications()
                character.additional_notes += f"\nModification request {version}: {modifications}"
                version += 1

                if version > 5:
                    print("Reached maximum iterations. Using latest version as final.")
                    character.final_image = image_path
                    break

        # After 5 versions, prompt user to select from the 5 images
        print("\nYou've reached the maximum number of image versions.")
        print("Please choose one of the following images as the final choice:")

        for i, img_path in enumerate(character.image_versions, 1):
            print(f"{i}: {img_path}")

        while True:
            try:
                choice = int(input(f"Enter the number (1–{len(character.image_versions)}) of the image you want to use: ").strip())
                if 1 <= choice <= len(character.image_versions):
                    character.final_image = character.image_versions[choice - 1]
                    print(f"Final image selected: {character.final_image}")
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def generate_character_image(self, character: Character, version: int) -> str:
        """Generate an image for the character using local"""
        # Create image prompt
        prompt = self.create_image_prompt(character)

        # Generate filename - simplified to just character name for final version
        safe_name = "".join(c for c in character.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if version == 1 and len(character.image_versions) == 0:
            # First version gets clean name
            filename = f"{safe_name}.png"
        else:
            # Subsequent versions get version number
            filename = f"{safe_name}_v{version}.png"

        filepath = os.path.join( self.working_folder, filename)

        print("Generating with local Stable Diffusion...")

        # Enhance prompt with local model optimization
        optimized_prompt = self.optimize_prompt_for_local_sd(prompt)

        with torch.no_grad():
            image = self.image_generator(
                prompt=optimized_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_INF_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                width=512,
                height=512
            ).images[0]

        image.save(filepath)
        print(f"Image saved locally: {filename}")

        return filepath 
    
    def generate_story_with_local_gemma(self, prompt:str, max_length: int = 512) -> str:
        """Generate story text using local Gemma model"""
        if not self.text_generator:
            raise Exception("Local Gemma model not loaded")
        
        try: 
            # Tokenize input
            inputs = self.gemma_tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # Generate text
            with torch.no_grad():
                outputs = self.text_generator.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.gemma_tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )

            # Decode generated text
            generated_text = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the original prompt from the output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text
        
        except Exception as e:
            print(f"Local Gemma generation failed: {e}")
            return "Story generation failed with local model."
        

    def create_story_characters(self, story_context: dict, max_characters: int = 5) -> List[Character]:
        """Create characters for a story using story context and local models""" 
        print(f" Creating story characters with context: {story_context['theme']}")

        # Generate character suggestions using local Gemma if available
        if self.text_generator:
            character_prompt = f"""
    Based on the story theme: {story_context['theme']}
    Style guidance: {story_context['guidance']}
    Create {max_characters} unique characters for this children's story. For each character, provide:
    - Name
    - Type (human, animal, fantasy creature, etc.)
    - Brief appearance description
    - Personality traits
    Characters:
        """
        try:
            suggestions = self.generate_story_with_local_gemma(character_prompt, max_length=1824)
            print(" P AI-generated character suggestions:")
            print (suggestions)
            print("\nYou can use these suggestions or create your own characters.")
        except Exception as e:
            print(f"ould not generate character suggestions: {e}")

        # Interactive character creation
        characters = []
        for i in range(max_characters):
            print(f"\n Creating Character {i + 1} of {max_characters}")
            print(f"Story context: {story_context['theme']}")

            character = self.create_character(i + 1)
            characters.append(character)

            # Ask if user wants to continue
            if i < max_characters - 1:
                continue_creation = input(f"InCreate another character? (y/n, {max_characters - i - 1} remaining): ").lower().strip()
            if continue_creation not in ['y', 'yes']:
                break

        return characters

    def optimize_prompt_for_local_sd(self, prompt: str) -> str:
        """Optimize prompt for local Stable Diffusion model"""
        # Add specific tokens that work well with SD v1.5
        optimization_tokens =[
            "storybook style"
        ]

        # Remove any tokens that might confuse the local model
        optimized = prompt

        # Add optimization tokens if not already present
        for token in optimization_tokens:
            if token.lower() not in optimized.lower():
                optimized += f", {token}" 

        return optimized

    def create_image_prompt(self, character: Character) -> str:
        """Create a detailed prompt for image generation"""
        prompt_parts = []

        # Basic description
        prompt_parts.append(f"A {character.character_type} character named {character.name}")

        # Gender
        if character.gender:
            prompt_parts.append(f"who is {character.gender}")

        # Appearance
        if character.appearance:
            prompt_parts.append(f"with the following appearance: {character.appearance}")
                            
        # Personality hints for visual style
        if character.personality:
            prompt_parts.append(f"The character has a {character.personality} personality, which should be reflected in their expression and pose")

        # Additional notes
        if character.additional_notes:
            prompt_parts.append(f"Additional details: {character.additional_notes}")
                                                        
        # Style instructions
        prompt_parts.append("Digital art style, high quality, detailed, for children storybook")

        return ". ".join(prompt_parts)
        

    def download_image(self, url: str, filepath: str):
        """Download image from URL and save to file"""
        response = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(response.content)


    def create_placeholder_image(self, filepath: str, character: Character, prompt: str):
        """Create a placeholder image file with character details"""

        # For demo purposes, create a text file with character details
        placeholder_content = f"""
    Character Image Placeholder
    ===========================

    Character Name: {character.name}
    Type: {character.character_type}
    Gender: {character.gender}
    Appearance: {character.appearance}
    Personality: {character.personality}
    Additional Notes: {character.additional_notes}
    Generated Prompt:
    {prompt}

    Note: This is a placeholder. In a real implementation with SD model, 
    this would be an actual generated image.
        """

        # Save as text file (in real implementation, this would be a PNG)
        text_filepath = filepath.replace('.png', '.txt')
        with open(text_filepath, "w") as f:
            f.write(placeholder_content)
        return text_filepath


    def get_user_feedback(self, character: Character, version: int) -> bool:
        """Get user feedback on generated image""" 
        print(f"\n Please review the generated image for {character.name} (version {version})")
        print(" Tip: Open the image file to view it")
        while True:
            feedback = input("Are you satisfied with this image? (yes/no): ").lower().strip()
            if feedback in ['yes', 'y']:
                return True
            elif feedback in ['no', 'n']:
                return False 
            else:
                print(" Please answer 'yes' or 'no'")
            
        
    def ask_for_modifications(self) -> str:
        """Ask user what modifications they want"""
        print("\n What would you like to change about the image?")
        print(" Be specific: colors, pose, facial expression, clothing, background, etc.")
        modifications = input("Modifications: ").strip()
        return modifications


    def show_final_summary(self):
        "Show summary of all created characters"
        print("\n" + "=" * 60)
        print(" CHARACTER CREATION COMPLETE!")
        print("=" * 60)
        for i, character in enumerate(self.characters, 1):
            print(f"(n{i}. {character.name}")
            print(f"Type: {character.character_type}")
            print(f"Gender: {character.gender}")
            print(f"Final Image: {character.final_image}")
            print(f"Total Versions: {len(character.image_versions)}")
            print(f"\n All character files saved in: {self.working_folder}")


    def save_session_data(self):
        """Save session data to JSON file"""
        session_data = {
        'session_id': self.session_folder,
        'created_at': self.now,
        'characters': [asdict(char) for char in self.characters]
        }

        json_path = os.path.join(self.working_folder, 'session_data.json')
        with open(json_path, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"Session data saved to: {json_path}")


def main():
    """Main function to run the character generator with local model support"""

    # Check for local model paths
    local_sd_path = os.getenv( 'LOCAL_SD_MODEL_PATH', 'stable-diffusion-v1-5-local')
    local_gemma_path = os.getenv("LOCAL_GEMMA_MODEL_PATH', 'local_gemma_model_path")

    print("Character Generation Agent - Local Model Edition")
    print("="* 60)
    print(f"Local SD Model Path: {local_sd_path}")
    print(f"Local Gemma Model Path: {local_gemma_path}")
    print()
    # Initialize agent with local models preferred
    agent = CharacterGeneratorAgent()
    agent.start_character_creation_session()


if __name__ == "__main__":
    main()
