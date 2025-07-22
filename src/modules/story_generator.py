from dotenv import load_dotenv
import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# Load environment variables
load_dotenv()
torch.set_default_dtype(torch.float16)  

# === Utility ===
# def extract_json_objects(content: str) -> list | dict:
#     """
#     Extracts and returns the valid JSON array or object found in markdown ```json code blocks.
#     """
#     print("Looking for ```json blocks...")
#     matches = re.findall(r"```json\s*(\{.*?\}|\[.*?\])\s*```", content, re.DOTALL)

#     if not matches:
#         raise ValueError("No valid JSON block found in the content.")

#     print(f"Found {len(matches)} JSON block(s).")
#     res_lst = []
#     for block in matches:
#         cleaned = re.sub(r",\s*(\}|\])", r"\1", block.strip())  # ✅ remove trailing commas
#         try:
#             parsed = json.loads(cleaned)
#             res_lst.append(parsed)
#         except json.JSONDecodeError as e:
#             print("⚠️ Skipping malformed block:\n", block[:300], "\n", str(e))

#     if not res_lst:
#         raise ValueError("All JSON blocks failed to parse.")

#     return res_lst[0] if len(res_lst) == 1 else res_lst

def extract_pagewise_json_objects(content: str) -> list:
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
            print(f"⚠️ Skipping malformed block #{i}:\n{block[:300]}\nError: {e}")

    if not results:
        raise ValueError("All JSON blocks failed to parse.")

    print(f"✅ Successfully extracted {len(results)} JSON objects.")
    print("Extracted object:\n", results if results else "None")
    return results

def extract_one_json_object(content: str) -> dict:
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
    print("Raw extracted block:\n", block)

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


# === Load Gemma Model ===
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model_gemma = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    local_files_only=True
)


# === Main Story Generator ===
def generate_story(story_theme, guidance, author_name, model="gemma-2b"):
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

    def call_model(messages):
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
            )
            return response.choices[0].message.content

        elif model == "gemma-2b":
            input_ids = tokenizer(prompt, return_tensors="pt").to(model_gemma.device)
            outputs = model_gemma.generate(
                **input_ids,
                do_sample=False,
                # temperature=0.7,
                max_new_tokens=int(1024 * 2),
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_output = outputs[0][input_ids["input_ids"].shape[1]:]
            return tokenizer.decode(generated_output, skip_special_tokens=True)

        else:
            raise ValueError(f"Unsupported model: {model}")

    # === Generate 15 Pages ===
    # story_prompt = (
    #     "Create a 15-page toddler story. "
    #     "Return ONLY a single JSON array (15 elements), wrapped in a single ```json block.\n"
    #     "If there's a name mentioned in the title of this book, please create a story of this animal.\n"
    #     "Remember each generated sentence of the story book should include conversation and thoughts, not pure description.\n"
    #     "Do NOT output separate blocks for each page.\n"
    #     "Format:\n```json\n[\n  { \"story_sentence\": \"...\", \"page_description\": \"...\" },\n  ...\n]\n```"
    # )
    # story_prompt = (
    #     "Create a 15-page toddler story. Each page must include:\n"
    #     "- story_sentence: a short, toddler-friendly sentence with emotion or dialogue\n"
    #     "- page_description: a rich, vivid visual description that matches the story\n\n"
    #     "Respond ONLY with a single JSON array containing exactly 15 objects. Each object must include keys story_sentence and page_description\n"
    #     "Important:\n"
    #     "- All property names and string values MUST be enclosed in double quotes.\n"
    #     "- Wrap the array inside ONE markdown block: ```json ... ```\n"
    #     "- DO NOT use '...' or truncate any values.\n"
    #     "- Ensure the array contains exactly 15 complete and fully formed story objects.\n"
    #     "- Each value of story_sentence won't have more than 12 tokens.\n"
    #     "- Each value of page_description won't have more than 15 tokens.\n"
    # )
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
    """
    "DO NOT include:\n"
    "- formatting examples or placeholder text\n"
    "- any ellipses like '...', or comments like '// more'\n"
    "- input prompt, explanations, commentary, or multiple code blocks\n"
    "- truncated arrays — ensure all 15 objects are fully generated with real content.\n\n"
    # "Respond by populating this template:\n```json\n[ {...}, {...}, ..., {...} ]\n```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": story_prompt}
    ]

    content = call_model(messages)
    print("Model output (batch):\n", content)

    pages_data = extract_pagewise_json_objects(content)

    if not isinstance(pages_data, list):
        raise ValueError("Expected a list of JSON objects for pages.")
    
    cleaned_pages = []
    for i, item in enumerate(pages_data):
        if "..." in item.get("story_sentence", "") or "..." in item.get("page_description", ""):
            print(f"⚠️ Skipping page {i+1} due to placeholder: {item}")
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

    content = call_model(messages)
    cover_result = extract_one_json_object(content)

    if not isinstance(cover_result, dict):
        raise ValueError("Expected a JSON object for cover descriptions.")

    cover_description_1 = cover_result["cover_description_1"].strip()
    cover_description_2 = cover_result["cover_description_2"].strip()

    return story_sentences, page_descriptions, cover_description_1, cover_description_2
