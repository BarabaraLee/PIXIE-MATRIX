from PIL import Image, ImageDraw, ImageFont

def create_cover_pages(title, author_name, subtitle, main_character_image):
    """Create the main cover page and secondary cover page."""
    # Main cover page
    cover_page = Image.new('RGB', (800, 1200), color=(255, 200, 200))
    d = ImageDraw.Draw(cover_page)
    d.text((50, 100), f"{title}", fill=(0, 0, 0))
    d.text((50, 200), f"{subtitle}", fill=(0, 0, 0))
    d.text((50, 300), f"Author: {author_name}", fill=(0, 0, 0))

    # Secondary cover page
    secondary_cover_page = Image.new('RGB', (800, 1200), color=(255, 230, 230))
    d = ImageDraw.Draw(secondary_cover_page)
    d.text((50, 100), f"Author: {author_name}", fill=(0, 0, 0))
    d.text((50, 200), "Illustrated by: {author_name}, assisted by GenAI", fill=(0, 0, 0))

    # Add main character image to secondary cover
    secondary_cover_page.paste(main_character_image, (600, 1000))

    return cover_page, secondary_cover_page
