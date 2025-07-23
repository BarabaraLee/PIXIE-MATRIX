# utils.py

"""Module to place text on images for storybook pages."""
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_image_type(img):
    """Check if the input is a PIL Image or a file path."""
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        try:
            return Image.open(img)
        except Exception as e:
            logger.error(f"Error opening image file {img}: {e}")
            return None