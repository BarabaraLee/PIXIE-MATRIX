import json 
from libs.constants import MAX_NEW_OUTPUT_TOKENS
from libs.utils import get_gemma_tokenizer_n_model
from pathlib import Path 
from typing import List, Dict, Any 
import re
import os
from datetime import datetime

class PixieStoryAgent:
    """Generate multiple story versions with character awareness"""

    def __init__(self, config_path: str="src/config/book_config.json"):
        self.config = self.load_config(config_path)
        self.characters = self.load_character_registry()
        self.output_dir = Path("src/intermediate_results/story_versions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.setup_llm(model=self.config["story_config"]["model"])

    def load_config(self, config_path: str) -> dict:
        try: 
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at path: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in config file: {e}")

    def load_character_registry(self) -> List[Dict[str, Any]]:
        path = Path("src/intermediate_results/character_registry.json")
        if not path.exists():
            print(f"Character registry not found at {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def setup_llm(self, model="Gemma-2b-it"):
        print(" Generating story versions")
        if model == "Gemma-2b-it":
            self.tokenizer, self.text_generator = get_gemma_tokenizer_n_model()
            self.model = model 
        else:
            raise ValueError("Only Gemma-2b-it is supported in this version.")
        print(f"Used LLM: {model}")

    
    def extract_pagewise_json_objects(self, content: str) -> list:
        """
        Extracts all individual JSON objects wrapped in ```json code blocks (per-page format)
        and returns them as a list of dictionaries.
        """
        print("Looking for individual ```json { ... } ``` blocks...")
        
        # Match each JSON object inside a ```json code block
        matches = re.findall(r"```json\s*({.*?})\s*```", content, re.DOTALL)

        if not matches:
            raise ValueError("No valid JSON blocks found in the content.")

        results = []
        for i, block in enumerate(matches, 1):
            # Clean: remove trailing commas (if any)
            cleaned = re.sub(r",\s*(\}|\])", r"\1", block.strip())

            # Ensure property names are quoted (defensive fix for LLMs)
            cleaned = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', cleaned)

            try:
                parsed = json.loads(cleaned)
                results.append(parsed)
            except json.JSONDecodeError as e:
                print(f" Skipping malformed block #{i}:\n{block[:300]}\nError: {e}")

        if not results:
            raise ValueError("All JSON blocks failed to parse.")

        print(f" Successfully extracted {len(results)} JSON objects.")
        print("Extracted object:\n" + (json.dumps(results, indent=2) if results else "[]"))
        return results
    
    def extract_one_json_object(self, content: str) -> dict:
        """
        Extract and parse a single JSON object from a markdown ```json block.

        Specifically designed for cover_prompt responses where the model is expected
        to return one JSON object with cover_description_1 and cover_description_2 keys.

        Cleans common LLM formatting issues:
        - Trailing commas before }.
        - Unquoted keys.

        Args:
            content (str): The full LLM response string.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If no valid object is found or if parsing fails.
        """
        print("Print the original genearted content:")
        print(content + '\n')
        print("Looking for ```json block with an object...")
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)

        if not matches:
            raise ValueError("No JSON object block found.")

        # pick the match which is does not include placeholder
        block = ""
        for match in matches:
            if "{...}" not in match:
                block = match
                break
        print("Raw extracted block:\n" + block)

        # Remove trailing commas before }
        cleaned = re.sub(r",\s*}", "}", block.strip())

        # Quote unquoted keys (e.g., cover_description_1: → "cover_description_1":)
        cleaned = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON object.\nError: {e}\nCleaned:\n{cleaned[:300]}")

        if not isinstance(parsed, dict):
            raise ValueError("Parsed result is not a JSON object.")

        return parsed
    
    
    def call_model(self, messages, model="Gemma-2b-it"):
        prompt = "\n".join([m["content"] for m in messages])

        if model == "gpt-3.5-turbo":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7, 
                top_p=0.95,         # samples from top 95% probability mass
                max_new_tokens=MAX_NEW_OUTPUT_TOKENS,
            )
            return response.choices[0].message.content

        elif model == "Gemma-2b-it":
            tokenizer, model_gemma = get_gemma_tokenizer_n_model()
            input_ids = tokenizer(prompt, return_tensors="pt").to(model_gemma.device)
            outputs = model_gemma.generate(
                **input_ids,
                do_sample=True, # True result in non-deterministic generation, sampling enabled
                temperature=0.85, # control randomness (0.7–1.0 range works well)
                top_p=0.95,
                max_new_tokens=MAX_NEW_OUTPUT_TOKENS,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_output = outputs[0][input_ids["input_ids"].shape[1]:]
            result = tokenizer.decode(generated_output, skip_special_tokens=True)
            return result

        else:
            raise ValueError(f"Unsupported model: {model}")
        
    # === Main Story Generator ===
    def generate_story(self, story_theme, guidance, author_name, model="Gemma-2b-it"):
        story_sentences = []
        page_descriptions = []

        system_prompt = (
            f"You are a storybook writer and illustration assistant.\n"
            f"Story Theme: {story_theme}\n"
            f"Guidance: {guidance}\n"
            f"Each story page should include:\n"
            "- A toddler-friendly sentence\n"
            "- An illustration description that builds consistently on prior pages\n"
            "Be visually detailed with setting, lighting, and character action.\n"
            "DO NOT use placeholder values like \"...\". Each object must be complete and contain real values.\n"
        )

        story_prompt = (    "Create a 15-page toddler story. Each page must include:\n"
        "- story_sentence: a short, toddler-friendly sentence with emotion or dialogue\n"
        "- page_description: a rich, vivid visual description that matches the story\n\n"
        "Respond ONLY with a single JSON array (wrapped in one ```json block) containing exactly 15 objects. Each object must include:\n"
        "- story_sentence\n"
        "- page_description\n\n"
        "Template of the returned JSON array:\n"
        """
        ```json
        [
        { "story_sentence": "real text", "page_description": "real description" },
        14 more objects like this (total 15)
        ]\n
        ```
        """
        """Both "story_sentence" and "page_description" have max length of 65 tokens.\n"""
        "DO NOT include:\n"
        "- formatting examples or placeholder text\n"
        "- any ellipses like '...', or comments like '// more'\n"
        "- input prompt, explanations, commentary, or multiple code blocks\n"
        "- truncated arrays — ensure all 15 objects are fully generated with real content.\n\n"
        # "Start your response directly with:\n```json\n["
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story_prompt}
        ]

        content = self.call_model(messages)
        print("Model output (batch):\n" + content)

        pages_data = self.extract_pagewise_json_objects(content)

        if not isinstance(pages_data, list):
            raise ValueError("Expected a list of JSON objects for pages.")
        
        cleaned_pages = []
        for i, item in enumerate(pages_data):
            if "..." in item.get("story_sentence", "") or "..." in item.get("page_description", ""):
                print(f" Skipping page {i+1} due to placeholder: {item}")
                continue
            cleaned_pages.append(item)

        pages_data = cleaned_pages

        story_sentences = [item["story_sentence"].strip() for item in pages_data]
        page_descriptions = [item["page_description"].strip() for item in pages_data]

        # === Generate Cover Descriptions ===
        cover_prompt = (
            f"Based on the story theme and guidance, generate 2 illustration descriptions for the book covers:\n"
            f"1. cover_description_1: Eye-catching cover for children’s book including author name: {author_name}\n"
            f"2. cover_description_2: A second cover with warm pastel background and main character in corner.\n"
            f"Include text: 'Author: {author_name}' and 'Illustrated by: {author_name}, assisted by GenAI'\n"
            "Return a JSON object with keys: cover_description_1, cover_description_2 wrapped in one ```json block."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cover_prompt}
        ]

        content = self.call_model(messages)
        cover_result = self.extract_one_json_object(content)

        if not isinstance(cover_result, dict):
            raise ValueError("Expected a JSON object for cover descriptions.")

        cover_description_1 = cover_result["cover_description_1"].strip()
        cover_description_2 = cover_result["cover_description_2"].strip()

        return story_sentences, page_descriptions, cover_description_1, cover_description_2
    

    def generate_story_versions(self, num_versions: int = 3):
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        output_dir = Path("story_collection") / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {num_versions} story versions for session: {session_id}")

        for i in range(1, num_versions + 1):
            print(f"\n--- Generating version {i} ---")
            # try:
            story_sentences, page_descriptions, cover1, cover2 = self.generate_story(
                story_theme=self.config["story_theme"],
                guidance=self.config["guidance"],
                author_name=self.config["author_name"],
                model=self.model
            )

            # Create a summary from the first and last sentence
            summary = {
                "summary": {
                    "beginning": story_sentences[0],
                    "ending": story_sentences[-1],
                    "page_count": len(story_sentences)
                }
            }

            output_data = {
                "cover_description_1": cover1,
                "cover_description_2": cover2,
                "story_sentences": story_sentences,
                "page_descriptions": page_descriptions,
                **summary
            }

            with open(output_dir / f"version_{i}.json", "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved version {i} to {output_dir / f'version_{i}.json'}")

            # except Exception as e:

            #     print(f"❌ Failed to generate version {i}: {e}")


if __name__ == "__main__":
    agent = PixieStoryAgent()
    agent.generate_story_versions()