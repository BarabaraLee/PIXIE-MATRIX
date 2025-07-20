import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from modules.story_generator import generate_story
from modules.illustration_generator import generate_illustrations
from modules.text_placer import place_text_on_images
from modules.epub_assembler import assemble_epub
from modules.cover_creator import create_cover_pages
from modules.sketch_to_prompt_generator import sketch_to_prompt_dict

def get_user_input():
    """Get user input for title, subtitle, author name, and optional guidance."""
    title = input("Enter the book title: ")
    subtitle = input("Enter the story theme (subtitle): ")
    author_name = input("Enter the author name: ")
    guidance = input("Enter any additional guidance (optional): ")
    return title, subtitle, author_name, guidance

def main():
    # Step 1: Get user input
    title, subtitle, author_name, guidance = get_user_input()

    # Step 1.5: Optional sketch layout guidance from up to 5 sketches
    use_sketch = input("Do you want to include 1–5 sketch images as layout guidance? (y/n): ").strip().lower()
    sketch_prompts_dict = {}

    if use_sketch == "y":
        max_sketches = 5
        try:
            num_sketches = int(input("How many sketches do you want to use? (1–5): ").strip())
            if not (1 <= num_sketches <= max_sketches):
                raise ValueError
        except ValueError:
            print(f"Invalid number. Defaulting to 1 sketch.")
            num_sketches = 1

        labeled_inputs = []
        for i in range(num_sketches):
            print(f"\n--- Sketch {i+1} ---")
            label = input("Label this sketch (e.g., 'scene1', 'layout2'): ").strip()
            path = input("Enter the path to your sketch image: ").strip()
            sketch_type = input("Describe the sketch type (e.g., room, street, landscape): ").strip()
            labeled_inputs.append((label, path, sketch_type))

        sketch_prompts_dict = sketch_to_prompt_dict(labeled_inputs)

        # Combine all prompts into one guidance string
        combined_guidance = "\n".join([f"[{label}] {prompt}" for label, prompt in sketch_prompts_dict.items()])

        if guidance:
            guidance += "\n" + combined_guidance
        else:
            guidance = combined_guidance

    # Step 2: Generate story sentences using combined guidance
    story_sentences = generate_story(subtitle, guidance)

    # Step 3: Generate illustrations
    illustrations = generate_illustrations(story_sentences)

    # Step 4: Place text on images
    pages = place_text_on_images(story_sentences, illustrations)

    # Step 5: Create cover pages
    cover_page, secondary_cover_page = create_cover_pages(title, author_name, subtitle, illustrations[0])

    # Step 6: Assemble EPUB
    assemble_epub(pages, cover_page, secondary_cover_page, author_name)

if __name__ == "__main__":
    main()
