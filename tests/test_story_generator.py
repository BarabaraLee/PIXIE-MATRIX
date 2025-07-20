import unittest
from unittest.mock import patch, Mock
from src.modules.story_generator import generate_story


class TestStoryGenerator(unittest.TestCase):

    @patch("openai.Completion.create")
    def test_generate_story_gpt(self, mock_completion_create):
        """Test the generate_story function with a mocked OpenAI API response."""
        mock_completion_create.return_value = {
            "choices": [
                {"text": "Once upon a time in a sunny tundra, a bear cub named Benny met a clever Arctic fox named Fifi."}
            ]
        }

        story_theme = "Sunny tundra adventures"
        guidance = "A bear cub and its animal friends explore the tundra."

        story_sentences = generate_story(story_theme, guidance, model="gpt-3.5-turbo")

        self.assertEqual(len(story_sentences), 1)
        self.assertIn("Once upon a time in a sunny tundra", story_sentences[0])

    @patch("groq.Groq.chat.completions.create")
    def test_generate_story_gemma(self, mock_groq_create):
        """Test the generate_story function with a mocked Groq (Gemma-7B) response."""
        # Setup mock Groq response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Once upon a time in a sunny tundra, a bear cub named Benny met his friends."))
        ]
        mock_groq_create.return_value = mock_response

        story_theme = "Sunny tundra adventures"
        guidance = "A bear cub and its animal friends explore the tundra."

        story_sentences = generate_story(story_theme, guidance, model="gemma-7b")

        self.assertEqual(len(story_sentences), 1)
        self.assertIn("Once upon a time in a sunny tundra", story_sentences[0])


if __name__ == "__main__":
    unittest.main()