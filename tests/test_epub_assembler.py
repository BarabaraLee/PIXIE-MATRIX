import unittest
import os
from PIL import Image
from src.modules.epub_assembler import assemble_epub
from ebooklib import epub

class TestEpubAssembler(unittest.TestCase):

    def setUp(self):
        # Create dummy images for cover and pages
        self.cover = Image.new("RGB", (1024, 1536), color="red")
        self.secondary_cover = Image.new("RGB", (1024, 1536), color="blue")
        self.pages = [
            Image.new("RGB", (1024, 1536), color="green"),
            Image.new("RGB", (1024, 1536), color="yellow")
        ]
        self.author_name = "Jane Doe"
        self.output_file = "toddler_book.epub"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_assemble_epub_creates_file(self):
        """Test if assemble_epub successfully creates an EPUB file."""
        assemble_epub(self.pages, self.cover, self.secondary_cover, self.author_name)
        self.assertTrue(os.path.exists(self.output_file))

    def test_assemble_epub_file_is_valid(self):
        """Test if the created EPUB file can be read by ebooklib."""
        assemble_epub(self.pages, self.cover, self.secondary_cover, self.author_name)
        book = epub.read_epub(self.output_file)

        # Check metadata
        self.assertEqual(book.get_metadata('DC', 'title')[0][0], "Toddler Book")
        self.assertEqual(book.get_metadata('DC', 'language')[0][0], "en")
        self.assertEqual(book.get_metadata('DC', 'creator')[0][0], self.author_name)

        # Check items in spine (1 secondary cover + 2 pages)
        spine_ids = [item[0] for item in book.spine if item[0] != 'nav']
        self.assertEqual(len(spine_ids), 3)

if __name__ == "__main__":
    unittest.main()
