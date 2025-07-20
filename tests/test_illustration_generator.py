import unittest
import os
import openai
from unittest.mock import patch, Mock
from src.modules.illustration_generator import generate_illustrations

# Suppress urllib3 warnings about OpenSSL compatibility
# warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

class TestGenerateIllustrations(unittest.TestCase):

    @patch("openai.Image.create")
    def test_generate_illustrations_dalle(self, mock_image_create):
        """Test if generate_illustrations works with DALL·E model."""
        story_sentences = [
            "A playful bear cub in a sunny tundra with an Arctic fox.",
            "The musk ox calf and snowy owl chick join the fun.",
            "The snowshoe hare hops around happily."
        ]

        # Mock the response from openai.Image.create
        mock_image_create.return_value = {
            'data': [{'url': 'http://example.com/fake_image.jpg'}]
        }

        illustrations = generate_illustrations(story_sentences, model="dall-e-3")
        self.assertEqual(len(illustrations), len(story_sentences))
        for url in illustrations:
            self.assertTrue(url.startswith("http"))

        self.assertEqual(mock_image_create.call_count, len(story_sentences))

    @patch("requests.post")
    def test_generate_illustrations_huggingface_sdxl(self, mock_post):
        """Test if generate_illustrations works with Hugging Face SDXL model."""
        story_sentences = [
            "A tiny frog sits on a lily pad in a peaceful pond.",
            "A duckling waddles through a field of wildflowers."
        ]

        # Set up mock response object
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'FAKE_IMAGE_BYTES'
        mock_post.return_value = mock_response

        # Create a temporary API token for the test
        os.environ["HUGGINGFACE_API_TOKEN"] = "hf_fake_token_for_testing"

        illustrations = generate_illustrations(story_sentences, model="huggingface-SDXL")
        self.assertEqual(len(illustrations), len(story_sentences))
        for file in illustrations:
            self.assertTrue(file.endswith(".png"))
            self.assertTrue(os.path.exists(file))

        # Cleanup generated test files
        for file in illustrations:
            if os.path.exists(file):
                os.remove(file)

        self.assertEqual(mock_post.call_count, len(story_sentences))

if __name__ == "__main__":
    unittest.main()