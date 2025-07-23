# cover_creator.py
"""Module to create main and secondary cover pages with custom layout."""

from PIL import Image, ImageDraw, ImageFont
from constants import HEIGHT, WIDTH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_text_size(text, font):
    """
    Returns (width, height) of the given text rendered in the specified font.
    Compatible with Pillow >=10.0, replacing draw.textsize.
    """
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

def create_cover_pages(title, author_name, subtitle):
    """Create the main cover page and secondary cover page with custom layout."""
    # Define image size
    logger.info(f"Creating cover pages with size: {WIDTH}x{HEIGHT}")
    logger.info(f"Pasting resized character image at {(WIDTH - 260, HEIGHT - 260)}")

    # Load default fonts (fallback if custom font is unavailable)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 72)
        subtitle_font = ImageFont.truetype(font_path, 48)
        author_font = ImageFont.truetype(font_path, 32)
    except Exception as e:
        logger.info(f"Error loading fonts: {e}, loading default font now.")
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        author_font = ImageFont.load_default()

    # === Main Cover Page ===
    cover_page = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 200, 200))
    draw = ImageDraw.Draw(cover_page)

    # Center title and subtitle at the top
    title_w, title_h = get_text_size(title, title_font)
    subtitle_w, subtitle_h = get_text_size(subtitle, subtitle_font)

    draw.text(((WIDTH - title_w) / 2, 120), title, fill=(0, 0, 0), font=title_font)
    draw.text(((WIDTH - subtitle_w) / 2, 120 + title_h + 30), subtitle, fill=(0, 0, 0), font=subtitle_font)

    # Author at bottom center
    author_text = f"Author: {author_name}"
    author_w, author_h = get_text_size(author_text, author_font)
    draw.text(((WIDTH - author_w) / 2, HEIGHT - author_h - 100), author_text, fill=(0, 0, 0), font=author_font)

    # === Secondary Cover Page ===
    secondary_cover_page = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 230, 230))
    draw2 = ImageDraw.Draw(secondary_cover_page)

    # Author & illustrator on left half
    text_block = (
        f"Author: {author_name}\n"
        f"Illustrated by: {author_name}, assisted by GenAI"
    )
    draw2.multiline_text((80, 150), text_block, fill=(0, 0, 0), font=subtitle_font, spacing=30)

    return cover_page, secondary_cover_page