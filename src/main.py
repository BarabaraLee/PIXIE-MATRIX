import sys
import os
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from modules.story_generator import generate_story
from modules.illustration_generator import generate_illustrations
from modules.text_placer import place_text_on_images
from modules.epub_assembler import assemble_epub
from modules.cover_creator import create_cover_pages
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def main():
    """Main function to generate a children's book EPUB.
    
    Sample JSON config:
    {
        "title": "My Book",
        "subtitle": "Sunny Tundra Adventure",
        "author_name": "Jane Doe",
        "guidance": "Use warm colors.",
        "sketch_map": {
            "cover1": { "path": "sketches/cover1.png", "pages": [1] },
            "cover2": { "path": "sketches/cover2.png", "pages": [2] },
            "sketch1": { "path": "sketches/sk1.png", "pages": [3,4,5,6] },
            "sketch2": { "path": "sketches/sk2.png", "pages": [7,8,9,10] },
            "sketch3": { "path": "sketches/sk3.png", "pages": [11,12,13,14,15] }
        }
    }

    Run with:
        make run CONFIG=config/book_config.json SKETCH_DIR=sketches/, or
        make run CONFIG=config/book_config.json SKETCH_DIR=""
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--sketch_dir", type=str, default="", help="Folder containing sketch images")
    args = parser.parse_args()
    logger.info(f"Using config file: {args.config}")
    logger.info(f"Using sketch directory: {args.sketch_dir}")

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)
    logger.info("Loaded config:", config)

    title = config["title"]
    subtitle = config["subtitle"]
    author_name = config["author_name"]
    story_theme = config["story_theme"] 
    guidance = config.get("guidance", "")
    sketch_map_raw = config.get("sketch_map", {})

    # Build sketch_map by prepending sketch_dir to file paths (if sketch_dir is provided)
    sketch_map = {}
    if args.sketch_dir:
        for label, val in sketch_map_raw.items():
            updated_path = os.path.join(args.sketch_dir, os.path.basename(val["path"]))
            sketch_map[label] = {"path": updated_path, "pages": val["pages"]}

    # Step 1: Generate story sentences, 
    # using: generate_story(story_theme, guidance, author_name, model="gemma-2b")
    story_sentences, page_descriptions, cover_description_1, cover_description_2 = \
        generate_story(story_theme, guidance, author_name=author_name)
    
    logger.info("Generated story sentences: ", story_sentences)
    logger.info("Generated page descriptions: ", page_descriptions)
    logger.info("Generated cover description 1: ", cover_description_1)
    logger.info("Generated cover description 2: ", cover_description_2)

    # Step 2: Generate illustrations using SDXL + ControlNet
    e2e_story_sents = [cover_description_1, cover_description_2] + page_descriptions
    illustrations = generate_illustrations(
        e2e_story_sents,
        model="huggingface-SDXL",
        sketch_map=sketch_map,
        title=title
    )

    # Step 3: Place text on images
    pages = place_text_on_images(story_sentences, illustrations)

    # Step 4: Create cover pages
    cover_page, secondary_cover_page = create_cover_pages(title, author_name, subtitle)

    # Step 5: Assemble EPUB
    assemble_epub(pages, cover_page, secondary_cover_page, author_name)

if __name__ == "__main__":
    main()
