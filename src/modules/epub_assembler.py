from ebooklib import epub

def assemble_epub(pages, cover_page, secondary_cover_page, author_name):
    """Assemble the pages and cover into an EPUB file."""
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier("id123456")
    book.set_title("Toddler Book")
    book.set_language("en")
    book.add_author(author_name)

    # Add cover pages
    book.set_cover("cover.jpg", cover_page.tobytes())

    # Add pages
    for i, page in enumerate(pages):
        chapter = epub.EpubHtml(title=f"Page {i+1}", file_name=f"page_{i+1}.xhtml", lang="en")
        chapter.content = f'<html><body><img src="page_{i+1}.jpg" /></body></html>'
        book.add_item(chapter)

    # Define Table of Contents and Spine
    book.toc = tuple(book.items)
    book.spine = ['nav'] + list(book.items)

    # Write to file
    epub.write_epub("toddler_book.epub", book)
