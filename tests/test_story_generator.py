import unittest
from unittest.mock import patch, Mock, MagicMock
import json
from src.modules.story_generator import generate_story

class TestGenerateStory(unittest.TestCase):

    @patch('src.modules.story_generator.AutoModelForCausalLM.from_pretrained')
    @patch('src.modules.story_generator.AutoTokenizer.from_pretrained')
    def test_generate_story_gemma(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tokenizer.return_value = {'input_ids': MagicMock()}
        mock_tokenizer.decode.return_value = json.dumps({
            "story_sentence": "Benny Bear woke up with a smile.",
            "page_description": "A cozy bedroom scene with Benny Bear stretching awake."
        })

        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model.generate.return_value = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        story_sentences, page_descriptions, cover1, cover2 = generate_story(
            story_theme="Friendship Adventure",
            guidance="Make it playful and cheerful",
            author_name="Jane Doe",
            model="gemma-2b"
        )

        self.assertEqual(len(story_sentences), 15)
        self.assertEqual(len(page_descriptions), 15)

    @patch('src.modules.story_generator.OpenAI')
    def test_generate_story_gpt35(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "story_sentence": "Benny Bear woke up with excitement.",
                "page_description": "Bright morning, Benny Bear excitedly gets out of bed."
            })))]
        ) for _ in range(15)] + [
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "cover_description_1": "Cheerful Benny Bear with friends on playful adventure. Author: Jane Doe",
                "cover_description_2": "Warm pastel background, Benny in bottom-right corner. Text: 'Author: Jane Doe', 'Illustrated by: Jane Doe, assisted by GenAI'"
            })))])
        ]

        story_sentences, page_descriptions, cover1, cover2 = generate_story(
            story_theme="Playful Day",
            guidance="Bright and joyful",
            author_name="Jane Doe",
            model="gpt-3.5-turbo"
        )

        self.assertEqual(len(story_sentences), 15)
        self.assertEqual(len(page_descriptions), 15)
        self.assertEqual(cover1, "Cheerful Benny Bear with friends on playful adventure. Author: Jane Doe")
        self.assertIn("Illustrated by: Jane Doe", cover2)

if __name__ == "__main__":
    unittest.main()
