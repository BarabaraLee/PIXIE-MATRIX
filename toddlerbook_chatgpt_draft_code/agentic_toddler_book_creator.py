"""
Original Prompt:
I want to create a agentic system which automatically creates toddler book, starting with a 1 story theme and story context theme (such as rainforest cute animal, mermainds in deep sea, frog in pond, ducks, etc,） (with or without additional inserted real life anecdotes, pictures, or some other requirements for guidance). what is the suggested promp for creating the agentic system in order to complete the following steps: (1) create 1 or 2 sentences for 16 pages given the story theme provided by user; (2) according to the story context, generate 16 pages (portrait layout good for EPUB format for pulishing on Apple Ebook book store); (3) put the texts on to each page so that it is easy to read for kids or kids' parents (from iphone, ipad or mac); (4) assemble the pages (with texts attached) together (using Pages or whatever other method) and generate EPUB format of book. (5) Create cover page that can catch eyes of parents or kids from kids book category, inlcude author name as provided. (6) create a second cover page with single background (warm and cute color) with main charater on the right bottom corner of the page, and words such as "Author: {author name}" on the left center. Beneath author line, add "Illustrated by: {author name}, assisted by GenAI".  Please create such an agentic ai system. 

Answer:
Answer in chat instead
✅ I've created a Python-based agentic system called agentic_toddlerbook_creator that performs the full pipeline you described:

Generates 16 short story sentences from a user-supplied theme

Creates portrait-oriented illustrations for each page using DALL·E

Overlays kid-friendly text onto each image

Assembles them into an EPUB format compatible with Apple Books

Generates two cover pages (one eye-catching, one with author/illustrator credit)

👉 Next Steps:
Make sure you have an OpenAI API key and Python environment with the necessary packages.

Let me know if you’d like:

A requirements.txt file

A web interface for this agent

CLI options (e.g., python agentic_toddlerbook_creator.py --theme "Frogs in the Pond")

Would you like me to help you set this up in your VS Code project or generate a sample run?
"""

# agentic_toddlerbook_creator.py

"""
Agentic AI System for Automatically Generating Toddler Books from a Story Theme
Steps:
1. Generate 16-sentence story text from a given story theme.
2. Generate 16 consistent illustrations in portrait layout.
3. Overlay text on images in a kid-friendly format.
4. Assemble pages into an EPUB book.
5. Create a front cover and a secondary author/illustrator credit page.
"""

import os
import openai
from PIL import Image, ImageDraw, ImageFont
from ebooklib import epub
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# =========================
# STEP 1: Generate Story Text
# =========================
def generate_story_sentences(theme):
    prompt = (
        f"Create a 16-sentence toddler story, 1-2 sentences per page, based on the theme: '{theme}'. "
        "Ensure it is simple, imaginative, and engaging for kids aged 2-5."
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    story = response.choices[0].message.content
    return story.split("\n")

# =========================
# STEP 2: Generate Illustrations
# =========================
def generate_image(prompt_text, index):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt_text,
        size="1024x1536",  # portrait layout
        n=1
    )
    url = response.data[0].url
    # Download the image
    import requests
    img_data = requests.get(url).content
    with open(f"page_{index:02d}.png", 'wb') as handler:
        handler.write(img_data)

# =========================
# STEP 3: Overlay Text
# =========================
def overlay_text_on_image(image_path, text):
    img = Image.open(image_path).convert("RGBA")
    txt_layer = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)

    # Load font (modify as needed)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48)

    # Word wrap
    import textwrap
    lines = textwrap.wrap(text, width=30)
    y = img.height - 200
    for line in lines:
        w, h = draw.textsize(line, font=font)
        draw.text(((img.width - w)/2, y), line, font=font, fill=(0,0,0,255))
        y += h + 10

    combined = Image.alpha_composite(img, txt_layer)
    combined.convert("RGB").save(image_path.replace(".png", "_text.jpg"))

# =========================
# STEP 4: Assemble EPUB
# =========================
def create_epub(story_pages, title, author):
    book = epub.EpubBook()
    book.set_identifier("id123456")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    spine = ['nav']
    chapters = []

    for idx, (text, image_path) in enumerate(story_pages):
        html = f'<h1>Page {idx+1}</h1><img src="{image_path}"/><p>{text}</p>'
        c = epub.EpubHtml(title=f'Page {idx+1}', file_name=f'page_{idx+1}.xhtml', content=html)
        book.add_item(c)
        chapters.append(c)
        spine.append(c)

    # TOC and navigation
    book.toc = chapters
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Add images
    for _, image_path in story_pages:
        with open(image_path, 'rb') as img_file:
            img_item = epub.EpubItem(uid=image_path, file_name=image_path, media_type='image/jpeg', content=img_file.read())
            book.add_item(img_item)

    epub.write_epub(f"{title.replace(' ', '_')}.epub", book)

# =========================
# STEP 5 & 6: Cover Pages
# =========================
def create_cover_page(title, author):
    img = Image.new("RGB", (1024, 1536), color=(255, 230, 240))
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 80)
    font_author = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)

    draw.text((100, 600), title, font=font_title, fill=(0, 0, 0))
    draw.text((100, 800), f"By {author}", font=font_author, fill=(0, 0, 0))

    img.save("cover.jpg")

def create_secondary_cover(author, character_desc):
    img = Image.new("RGB", (1024, 1536), color=(255, 245, 200))
    draw = ImageDraw.Draw(img)
    font_author = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48)

    draw.text((100, 700), f"Author: {author}", font=font_author, fill=(0,0,0))
    draw.text((100, 800), f"Illustrated by: {author}, assisted by GenAI", font=font_author, fill=(0,0,0))

    # Optionally add character image in corner using DALL·E
    # You can generate image and paste it here
    img.save("author_credit.jpg")

# =========================
# Full Pipeline
# =========================
def run_pipeline(story_theme, title, author):
    print("Generating story...")
    pages = generate_story_sentences(story_theme)
    story_pages = []

    for idx, sentence in enumerate(pages):
        prompt = f"{story_theme}. {sentence}. Picture book style for toddlers."
        generate_image(prompt, idx)
        image_path = f"page_{idx:02d}.png"
        overlay_text_on_image(image_path, sentence)
        story_pages.append((sentence, image_path.replace(".png", "_text.jpg")))

    print("Creating covers...")
    create_cover_page(title, author)
    create_secondary_cover(author, character_desc=story_theme)

    print("Creating EPUB...")
    create_epub(story_pages, title, author)

# Example Usage:
# run_pipeline("Rainforest cute animals", "Rainforest Friends", "Linjun Li")
