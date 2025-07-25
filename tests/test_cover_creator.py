import unittest
from PIL import Image
import sys
import os

# Add the path to src/modules to sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "modules")))

from src.modules.cover_creator import create_cover_pages
from libs.constants import WIDTH, HEIGHT


class TestCoverCreator(unittest.TestCase):

    def test_create_cover_pages_returns_images(self):
        title = "My Book"
        author = "Test Author"
        subtitle = "A Fun Adventure"

        cover, secondary = create_cover_pages(title, author, subtitle)

        # Test type
        self.assertIsInstance(cover, Image.Image)
        self.assertIsInstance(secondary, Image.Image)

        # Test dimensions
        self.assertEqual(cover.size, (WIDTH, HEIGHT))
        self.assertEqual(secondary.size, (WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main()