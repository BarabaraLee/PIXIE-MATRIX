import openai

def generate_story(story_theme, guidance):
    """Generate 16 sentences for the story based on the theme and guidance."""
    openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

    prompt = f"Create a 16-sentence story for a toddler book about {story_theme}. {guidance}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )

    story = response.choices[0].text.strip()
    return story.split("\n")
