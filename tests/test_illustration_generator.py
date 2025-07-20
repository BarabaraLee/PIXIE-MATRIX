import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from src.modules.illustration_generator import generate_illustrations

class TestIllustrationGenerator(unittest.TestCase):

    @patch('src.modules.illustration_generator.openai.Image.create')
    def test_generate_illustrations_dalle(self, mock_openai_create):
        mock_openai_create.return_value = {
            'data': [{'url': 'https://fakeurl.com/image.png'}]
        }

        sentences = ["Benny Bear wakes up happily.", "Benny Bear meets Finn Fox."]

        result = generate_illustrations(sentences, model="dall-e-3")

        self.assertEqual(len(result), 2)
        self.assertTrue(all(url == 'https://fakeurl.com/image.png' for url in result))

    @patch('src.modules.illustration_generator.cv2.imread')
    @patch('src.modules.illustration_generator.cv2.Canny')
    @patch('src.modules.illustration_generator.StableDiffusionControlNetPipeline.from_pretrained')
    @patch('src.modules.illustration_generator.ControlNetModel.from_pretrained')
    def test_generate_illustrations_sdxl(
        self, mock_controlnet, mock_pipeline, mock_canny, mock_imread
    ):
        # Mock cv2.imread to return a fake numpy array image
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8) * 255
        # Mock cv2.Canny to return a fake edge-detected image
        mock_canny.return_value = np.ones((512, 512), dtype=np.uint8) * 255

        mock_pipe_instance = MagicMock()
        mock_image = MagicMock(spec=Image.Image)
        mock_pipe_instance.return_value.images = [mock_image]
        mock_pipeline.return_value.to.return_value = mock_pipe_instance

        sentences = ["Benny Bear plays with Finn Fox.", "Benny Bear naps in the meadow."]
        sketch_map = {
            "sketch1": {"path": "path/to/sketch1.png", "pages": [1]},
        }

        result = generate_illustrations(sentences, model="huggingface-SDXL", sketch_map=sketch_map)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "page_01_sdxl.png")
        self.assertEqual(result[1], "page_02_sdxl.png")
        self.assertTrue(mock_image.save.called)

if __name__ == "__main__":
    unittest.main()
