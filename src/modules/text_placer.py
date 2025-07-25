# text_placer.py
"""Module to place text on images for storybook pages."""
from PIL import ImageDraw, ImageFont
from libs.constants import HEIGHT, WIDTH
from libs.utils import check_image_type
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_text_with_bg(draw, position, text, font, text_fill, bg_fill, alpha=128):
    """Draw text with a semi-transparent background behind it."""
    bbox = draw.textbbox(position, text, font=font)
    bg_color = bg_fill[:3] + (alpha,)
    draw.rectangle(bbox, fill=bg_color)
    draw.text(position, text, font=font, fill=text_fill)

def wrap_text(text, font, draw, max_width):
    """Split text into lines so each fits within max_width."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test_line = current + (" " if current else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def place_text_on_images(story_sentences, illustrations):
    """Place wrapped text at the middle bottom of each illustration image,
    with semi-transparent background, black text, and proper centering.
    """
    pages = []
    page_num = len(story_sentences)
    margin = 40
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 22)
    max_text_width = WIDTH - 80

    for sentence, img in zip(story_sentences[:page_num], illustrations[2:2+page_num]):
        img = check_image_type(img)
        if img is None:
            logger.error(f"Invalid image type for {img}. Skipping.")
            continue

        d = ImageDraw.Draw(img)
        lines = wrap_text(sentence, font, d, max_text_width)
        # Compute total height of all lines
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = d.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
            line_widths.append(bbox[2] - bbox[0])
        total_height = sum(line_heights) + (len(lines) - 1) * 4
        y = HEIGHT - total_height - margin
        for i, (line, lh, lw) in enumerate(zip(lines, line_heights, line_widths)):
            x = (WIDTH - lw) // 2
            draw_text_with_bg(
                d,
                (x, y),
                line,
                font=font,
                text_fill=(0, 0, 0, 255),
                bg_fill=(255, 255, 255, 0),
                alpha=128
            )
            y += lh + 4  # 4 pixels spacing between lines
        pages.append(img)
    return pages

def place_titles_authors_on_covers(cover_page, secondary_cover_page, title, author_name, subtitle):
    cover_img = check_image_type(cover_page)
    secondary_img = check_image_type(secondary_cover_page)

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_title = ImageFont.truetype(font_path, 32)
    font_subtitle = ImageFont.truetype(font_path, 28)
    font_author = ImageFont.truetype(font_path, 22)

    draw_cover = ImageDraw.Draw(cover_img)
    draw_secondary = ImageDraw.Draw(secondary_img)

    max_title_width = WIDTH - 80
    max_subtitle_width = WIDTH - 120

    # --- Main cover ---
    # Wrap and place title
    title_lines = wrap_text(title, font_title, draw_cover, max_title_width)
    y = 40
    for line in title_lines:
        bbox = draw_cover.textbbox((0, 0), line, font=font_title)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = (WIDTH - lw) // 2
        draw_text_with_bg(draw_cover, (x, y), line, font_title, (0,0,0,255), (255,255,255,0), alpha=128)
        y += lh + 6  # 6px spacing

    # Wrap and place subtitle, right below last title line
    subtitle_lines = wrap_text(subtitle, font_subtitle, draw_cover, max_subtitle_width)
    for line in subtitle_lines:
        bbox = draw_cover.textbbox((0, 0), line, font=font_subtitle)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = (WIDTH - lw) // 2
        draw_text_with_bg(draw_cover, (x, y), line, font_subtitle, (0,0,0,255), (255,255,255,0), alpha=128)
        y += lh + 4

    # Place author at bottom center
    author_text = f"Author: {author_name}"
    author_lines = wrap_text(author_text, font_author, draw_cover, WIDTH - 80)
    # Compute total height for author lines
    line_heights = [draw_cover.textbbox((0, 0), line, font=font_author)[3] - draw_cover.textbbox((0, 0), line, font=font_author)[1] for line in author_lines]
    total_author_height = sum(line_heights) + (len(author_lines) - 1) * 4
    y_author = HEIGHT - total_author_height - 40
    for i, line in enumerate(author_lines):
        bbox = draw_cover.textbbox((0, 0), line, font=font_author)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = (WIDTH - lw) // 2
        draw_text_with_bg(draw_cover, (x, y_author), line, font_author, (0,0,0,255), (255,255,255,0), alpha=128)
        y_author += lh + 4

    # --- Secondary cover ---
    # Title at upper left (wrap)
    y = 40
    title2_lines = wrap_text(title, font_title, draw_secondary, max_title_width)
    for line in title2_lines:
        bbox = draw_secondary.textbbox((0, 0), line, font=font_title)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = 40
        draw_text_with_bg(draw_secondary, (x, y), line, font_title, (0,0,0,255), (255,255,255,0), alpha=128)
        y += lh + 6

    # Author at lower left (wrap)
    author2_lines = wrap_text(author_name, font_author, draw_secondary, WIDTH - 80)
    line_heights = [draw_secondary.textbbox((0, 0), line, font=font_author)[3] - draw_secondary.textbbox((0, 0), line, font=font_author)[1] for line in author2_lines]
    total_author_height = sum(line_heights) + (len(author2_lines) - 1) * 4
    y2 = HEIGHT - total_author_height - 40
    for i, line in enumerate(author2_lines):
        bbox = draw_secondary.textbbox((0, 0), line, font=font_author)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = 40
        draw_text_with_bg(draw_secondary, (x, y2), line, font_author, (0,0,0,255), (255,255,255,0), alpha=128)
        y2 += lh + 4

    return cover_img, secondary_img
