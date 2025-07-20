from dotenv import load_dotenv
import os
from openai import OpenAI
from groq import Groq

# Load environment variables from .env file
load_dotenv()

def generate_story(story_theme, guidance, model="gemma-7b"):
    """Generate 15 sentences for the story based on the theme and guidance.
    Args:
        story_theme (str): The theme of the story.
        guidance (str): Additional guidance for story generation.
        model (str): The model to use for generation ("gpt-3.5-turbo" or "gemma-7b"), "gemma-7b" being the default.

    Returns:
        list: A list of 15 sentences forming the story.
    """
    prompt = (
        f"Create a 15-sentence story for a toddler book about {story_theme}. {guidance} "
        "The background of the story should be a sunny tundra region. The story revolves around a male cub bear named Benny and his interactions with his adorable animal friends: an Arctic fox named Finn, a musk ox named Ollie, a snowy owl named Snowy, and a snowshoe hare named Hoppy. "
        "Ensure the story is imaginative, heartwarming, and suitable for toddlers aged 2-5. Each character should have a unique and lovely personality that shines through their actions."
    )

    story = ""
    if model == "gpt-3.5-turbo":

        # Ensure the API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        # Start of story generation
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        story = response.choices[0].message.content.strip().split("\n")

    elif model == "gemma-7b":
        # Ensure the API key is set
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")

        # Start of story generation
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="gemma-7b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        story = response.choices[0].message.content.strip().split("\n")
        
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    return story
