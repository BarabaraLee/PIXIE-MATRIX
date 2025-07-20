from dotenv import load_dotenv
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import re

# Load environment variables
load_dotenv()


def extract_json_string(content: str) -> str:
    """
    Extract the final valid JSON object from a markdown code block labeled ```json.
    """
    print("Looking for ```json blocks...")
    matches = re.findall(r"```json\s*({.*?})\s*```", content, re.DOTALL)

    if not matches:
        raise ValueError("No valid JSON block found in the content.")

    # Take the last match
    json_candidate = matches[-1].strip()

    # print("Extracted JSON:\n", json_candidate)
    return json_candidate


# Load Gemma model
tokenizer = AutoTokenizer.from_pretrained(
    "/Users/linjunli/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad",
    local_files_only=True
    )
model_gemma = AutoModelForCausalLM.from_pretrained(
    "/Users/linjunli/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
    )


def generate_story(story_theme, guidance, author_name, model="gemma-2b"):
    """
    Generate a 15-sentence toddler story, 15 image descriptions, and 2 cover descriptions.
    """
    story_sentences = []
    page_descriptions = []

    # Shared system prompt
    system_prompt = (
        f"You are a storybook writer and illustration assistant.\n"
        f"Story Theme: {story_theme}\n"
        f"Guidance: {guidance}\n"
        f"Each story page should include:\n"
        "- A toddler-friendly sentence\n"
        "- An illustration description that builds consistently on prior pages\n"
        "Be visually detailed with setting, lighting, and character action.\n"
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
                    do_sample=True,             # ← enable sampling
                    temperature=0.7,            # ← now valid
                    max_new_tokens=512          # ← recommended over max_length
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        else:
            raise ValueError(f"Unsupported model: {model}")

    # === Generate 15 story pages ===
    for page_num in range(15):
        history = "\n".join([f"Page {i+1}: {s}" for i, s in enumerate(story_sentences)])

        user_prompt = (
            f"Now generate page {page_num+1}.\n"
            f"{'Here is what happened so far:\n' + history if history else ''}\n"
            """
            Respond ONLY with a single JSON object wrapped in a markdown block like:

            ```json
            {
            "story_sentence": "...",
            "page_description": "..."
            }
            ```
            """
            "Do NOT return multiple JSON objects.\n"
            "Do NOT return the format template.\n"
            """Do Not return any JSON value as "..." or '...' for the provided keys, they must be actual text values.\n"""
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        content = call_model(messages)
        print(f"**************************************\nModel output for page {page_num+1}:\n{content}\n**************************************\n")
        json_str = extract_json_string(content)

        print(f"Failed to parse output for page {page_num+1}:\n{content}")
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:   
            print("❌ Failed to decode JSON:")
            print(content)
            raise e
        
        story_sentences.append(parsed["story_sentence"].strip())
        page_descriptions.append(parsed["page_description"].strip())

    # === Generate cover descriptions ===
    cover_prompt = (
        f"Based on the story theme and guidance, generate 2 illustration descriptions for the book covers:\n"
        f"1. cover_description_1: Eye-catching cover for children’s book including author name: {author_name}\n"
        f"2. cover_description_2: A second cover with warm pastel background and main character in left-right corner. Include text: 'Author: {author_name}' and 'Illustrated by: {author_name}, assisted by GenAI'\n"
        "Return JSON with keys: cover_description_1, cover_description_2."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cover_prompt}
    ]

    content = call_model(messages)
    json_str = extract_json_string(content)

    try:
        cover_result = json.loads(json_str)
        cover_description_1 = cover_result["cover_description_1"].strip()
        cover_description_2 = cover_result["cover_description_2"].strip()
    except Exception as e:
        raise ValueError(f"Failed to parse cover descriptions:\n{content}\n{e}")

    return story_sentences, page_descriptions, cover_description_1, cover_description_2