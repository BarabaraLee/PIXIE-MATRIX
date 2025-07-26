from pathlib import Path
import cv2
from libs.constants import GUIDANCE_SCALE, HEIGHT, NUM_INF_STEPS, WIDTH
import numpy as np
from PIL import Image
import os

def generate_cover_image(prompt: str, character_names: list, pipeline, character_image_root: Path, output_path: Path):
    selected_images = []
    for char in character_names:
        folder = character_image_root / char
        images = sorted(folder.glob(f"{char}_pose*.png"))
        if not images:
            raise FileNotFoundError(f"No poses found for {char} in {folder}")
        print(f"Select pose for cover for character '{char}':")
        for idx, img in enumerate(images):
            print(f"{idx+1}. {img.name}")
        while True:
            try:
                choice = int(input("Enter pose number: "))
                selected_img = cv2.imread(str(images[choice - 1]))
                selected_images.append(selected_img)
                break
            except Exception as e:
                print(f"Invalid input: {e}")

    # Resize and concatenate
    target_height = 512
    resized = []
    for img in selected_images:
        h, w = img.shape[:2]
        scale = target_height / h
        resized_img = cv2.resize(img, (int(w * scale), target_height))
        resized.append(resized_img)
    composite = np.hstack(resized)

    # Convert to canny edge
    gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_pil = Image.fromarray(canny).convert("RGB")

    # Generate image
    result = pipeline(
        prompt=prompt,
        image=canny_pil,
        num_inference_steps=NUM_INF_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH
    ).images[0]
    result.save(output_path)
    print(f"✅ Saved cover image to {output_path}")