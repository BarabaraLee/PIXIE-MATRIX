import sys
import os
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from modules.story_generator import generate_story
from modules.illustration_generator import generate_illustrations
from modules.text_placer import place_text_on_images, place_titles_authors_on_covers
from modules.epub_assembler import assemble_epub
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def save_intermediate(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_intermediate(filename):
    with open(filename, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--sketch_dir", type=str, default="", help="Folder containing sketch images")
    parser.add_argument("--phase", choices=["gen", "assemble"], default="gen",
                        help="Which phase to run: 'gen' (steps 1+2), 'assemble' (steps 3+4+5)")
    parser.add_argument("--intermediate_file", type=str, default="src/intermediate_results/intermediate.json",
                        help="Path to store/load intermediate results")
    args = parser.parse_args()

    logger.info(f"Phase: {args.phase}")
    logger.info(f"Intermediate file: {args.intermediate_file}")

    if args.phase == "gen":
        # Run Steps 1+2
        assert args.config, "--config required for gen phase"

        logger.info(f"Using config file: {args.config}")
        logger.info(f"Using sketch directory: {args.sketch_dir}")

        with open(args.config, "r") as f:
            config = json.load(f)

        title = config["title"]
        subtitle = config["subtitle"]
        author_name = config["author_name"]
        story_theme = config["story_theme"]
        guidance = config.get("guidance", "")
        sketch_map_raw = config.get("sketch_map", {})

        sketch_map = {}
        if args.sketch_dir:
            for label, val in sketch_map_raw.items():
                updated_path = os.path.join(args.sketch_dir, os.path.basename(val["path"]))
                sketch_map[label] = {"path": updated_path, "pages": val["pages"]}

        # Step 1: Generate story sentences
        story_sentences, page_descriptions, cover_description_1, cover_description_2 = \
            generate_story(story_theme, guidance, author_name=author_name)
        
        logger.info(f"Generated story sentences: {story_sentences}")
        logger.info(f"Generated page descriptions: {page_descriptions}")

        # Step 2: Generate illustrations
        e2e_story_sents = [cover_description_1, cover_description_2] + page_descriptions
        illustrations = generate_illustrations(
            e2e_story_sents,
            model="huggingface-SDXL",
            sketch_map=sketch_map,
            title=title
        )

        # Save all results to intermediate file
        intermediate = {
            "story_sentences": story_sentences,
            "illustrations": illustrations,
            "title": title,
            "author_name": author_name,
            "subtitle": subtitle
        }
        save_intermediate(args.intermediate_file, intermediate)
        logger.info(f"Saved intermediate results to {args.intermediate_file}")

    elif args.phase == "assemble":
        # Run Steps 3+4+5
        data = load_intermediate(args.intermediate_file)
        logger.info(f"Loaded intermediate data from {args.intermediate_file}")

        story_sentences = data["story_sentences"]
        illustrations = data["illustrations"]
        title = data["title"]
        author_name = data["author_name"]
        subtitle = data["subtitle"]

        # Step 3: Place text on images
        page_imgs = place_text_on_images(story_sentences, illustrations)
        cover_page = illustrations[0]
        secondary_cover_page = illustrations[1]

        # Step 4: Create cover pages
        cover_img, secondary_img = place_titles_authors_on_covers(cover_page, secondary_cover_page, title, author_name, subtitle)

        # Step 5: Assemble EPUB
        assemble_epub(page_imgs, cover_img, secondary_img, author_name)
        logger.info("EPUB assembled.")

if __name__ == "__main__":
    main()
