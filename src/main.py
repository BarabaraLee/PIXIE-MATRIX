from modules.input_module import get_user_input
from modules.story_generator import generate_story
from modules.illustration_generator import generate_illustrations
from modules.text_placer import place_text_on_images
from modules.epub_assembler import assemble_epub
from modules.cover_creator import create_cover_pages

def main():
    # Step 1: Get user input
    story_theme, author_name, guidance = get_user_input()

    # Step 2: Generate story sentences
    story_sentences = generate_story(story_theme, guidance)

    # Step 3: Generate illustrations
    illustrations = generate_illustrations(story_sentences)

    # Step 4: Place text on images
    pages = place_text_on_images(story_sentences, illustrations)

    # Step 5: Create cover pages
    cover_page, secondary_cover_page = create_cover_pages(story_theme, author_name, illustrations[0])

    # Step 6: Assemble EPUB
    assemble_epub(pages, cover_page, secondary_cover_page, author_name)

if __name__ == "__main__":
    main()
