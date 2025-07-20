# To run from project root (for all project): python -m unittest discover -s tests -p "test_*.py"
# To run this file only: python -m unittest tests/test_cover_creator.py

import unittest
from PIL import Image
from src.modules.cover_creator import create_cover_pages


class TestCreateCoverPages(unittest.TestCase):

    def test_create_cover_pages_dimensions_and_types(self):
        # Arrange
        title = "My Toddler Book"
        author_name = "Jane Doe"
        subtitle = "A Happy Day"

        # Act
        cover_page, secondary_cover_page = create_cover_pages(title, author_name, subtitle)

        # Assert: Check output types
        self.assertIsInstance(cover_page, Image.Image)
        self.assertIsInstance(secondary_cover_page, Image.Image)

        # Assert: Check dimensions
        expected_size = (1024, 1536)
        self.assertEqual(cover_page.size, expected_size)
        self.assertEqual(secondary_cover_page.size, expected_size)

        # Assert: Check modes
        self.assertEqual(cover_page.mode, 'RGB')
        self.assertEqual(secondary_cover_page.mode, 'RGB')


if __name__ == "__main__":
    unittest.main()
