# text_placer.py
"""Module to place text on images for storybook pages."""
from PIL import ImageDraw

def place_text_on_images(story_sentences, illustrations):
    """Place text on each illustration image."""
    pages = []
    for sentence, img in zip(story_sentences[:15], illustrations[2:17]):  # Limit to 15 pages
        d = ImageDraw.Draw(img)
        d.text((50, 1050), sentence, fill=(0, 0, 0))  # Place text near the bottom
        pages.append(img)
    return pages
