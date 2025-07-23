from ebooklib import epub
from PIL import Image
import io
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def image_to_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

def assemble_epub(pages, cover_page, secondary_cover_page, author_name):
    """Assemble the pages and cover into an EPUB file."""
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier("id123456")
    book.set_title("Toddler Book")
    book.set_language("en")
    book.add_author(author_name)

    # Add main cover image
    book.set_cover("cover.jpg", image_to_bytes(cover_page))

    # Save secondary cover and story pages as XHTML chapters
    chapter_items = []

    # Add secondary cover page
    secondary_chapter = epub.EpubHtml(title="About the Author", file_name="cover2.xhtml", lang="en")
    secondary_chapter.content = '<html><body><img src="cover2.jpg" /></body></html>'
    book.add_item(secondary_chapter)

    # Store the image in the EPUB as a separate image file
    cover2_img = epub.EpubItem(
        uid="cover2.jpg",
        file_name="cover2.jpg",
        media_type="image/jpeg",
        content=image_to_bytes(secondary_cover_page)
    )
    book.add_item(cover2_img)
    chapter_items.append(secondary_chapter)

    # Add story pages
    for i, page in enumerate(pages):
        chapter = epub.EpubHtml(title=f"Page {i+1}", file_name=f"page_{i+1}.xhtml", lang="en")
        chapter.content = f'<html><body><img src="page_{i+1}.jpg" /></body></html>'
        book.add_item(chapter)

        # Embed the page image
        img_data = image_to_bytes(page)
        image_item = epub.EpubItem(
            uid=f"page_{i+1}.jpg",
            file_name=f"page_{i+1}.jpg",
            media_type="image/jpeg",
            content=img_data
        )
        book.add_item(image_item)

        chapter_items.append(chapter)

    # TOC and spine
    book.toc = tuple(chapter_items)
    book.spine = ['nav'] + chapter_items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Write to file
    epub.write_epub("toddler_book.epub", book)
    logger.info("EPUB file 'toddler_book.epub' created successfully.")