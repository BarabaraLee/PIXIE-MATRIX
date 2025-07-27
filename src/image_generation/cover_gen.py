from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from libs.constants import GUIDANCE_SCALE, HEIGHT, NEGATIVE_PROMPT, NUM_INF_STEPS, WIDTH
from libs.utils import remove_white_background, position_words, describe_character_appearance

def generate_cover_image(prompt: str, character_names: list, pipeline, character_image_root: Path, output_path: Path, layout_order: list = None):
    # Determine which characters are used
    used_characters = layout_order if layout_order else character_names

    selected_images = []
    for char in used_characters:
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
                img = cv2.imread(str(images[choice - 1]))
                img_rgba = remove_white_background(img)
                selected_images.append(img_rgba)
                break
            except Exception as e:
                print(f"Invalid input: {e}")

    # Create RGBA canvas
    canvas = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    num_images = len(selected_images)
    target_height = HEIGHT // 4
    gap = WIDTH // (num_images + 1)
    current_x = gap
    midpoint = num_images // 2

    for i, img in enumerate(selected_images):
        if num_images >= 2:
            if (num_images % 2 == 0 and i < midpoint) or (num_images % 2 == 1 and i < midpoint):
                img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        scale = target_height / h
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y_offset = (HEIGHT - new_h) // 2

        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            canvas[y_offset:y_offset+new_h, current_x:current_x+new_w, c] = (
                alpha * resized[:, :, c] +
                (1 - alpha) * canvas[y_offset:y_offset+new_h, current_x:current_x+new_w, c]
            )
        canvas[y_offset:y_offset+new_h, current_x:current_x+new_w, 3] = (
            np.maximum(canvas[y_offset:y_offset+new_h, current_x:current_x+new_w, 3], resized[:, :, 3])
        )
        current_x += new_w + gap

    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(rgb_canvas, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_pil = Image.fromarray(canny).convert("RGB")

    # Add spatial layout hint and character appearance to prompt
    layout_hint = ""
    if layout_order and len(layout_order) > 1:
        layout_hint = ", ".join([f"{name} is on the {pos}" for name, pos in zip(layout_order, position_words(len(layout_order)))]) + ". "

    appearance_hint, negative_background_hint = describe_character_appearance(used_characters)
    full_prompt = f"{layout_hint}{appearance_hint} {prompt}"

    result = pipeline(
        prompt=full_prompt,
        negative_prompt=negative_background_hint + NEGATIVE_PROMPT,
        image=canny_pil,
        num_inference_steps=NUM_INF_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH
    ).images[0]
    result.save(output_path)
    print(f" Saved cover image to {output_path}")
