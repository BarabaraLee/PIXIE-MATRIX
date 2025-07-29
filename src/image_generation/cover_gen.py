from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from libs.constants import GUIDANCE_SCALE, HEIGHT, NEGATIVE_PROMPT, NUM_INF_STEPS, WIDTH, NUM_IMAGES
from libs.utils import pose_number, remove_white_background, position_words, describe_character_appearance


def resize_to_diagonal(img: np.ndarray, diag_target: int) -> np.ndarray:
    h, w = img.shape[:2]
    original_diag = (h ** 2 + w ** 2) ** 0.5
    scale = diag_target / original_diag
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def generate_cover_image(
    prompt: str, 
    character_names: list, 
    pipeline, 
    character_image_root: Path, 
    output_path: Path, 
    output_file: str,
    layout_order: list = None):
    # Determine which characters are used
    used_characters = layout_order if layout_order else character_names

    selected_images = []
    for char in used_characters:
        folder = character_image_root / char
        images = sorted(folder.glob(f"{char}_pose*.png"), key=pose_number)
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
    diag_target = int((HEIGHT ** 2 + WIDTH ** 2) ** 0.5 * 0.4)  # 🟢 Fix here

    # Calculate horizontal placement
    if num_images == 1:
        positions_x = [WIDTH // 2]
    elif num_images == 2:
        positions_x = [int(WIDTH * 0.25), int(WIDTH * 0.75)]
    else:
        margin = int(WIDTH * 0.1)
        step = (WIDTH - 2 * margin) // (num_images - 1)
        positions_x = [margin + i * step for i in range(num_images)]

    for i, img in enumerate(selected_images):
        # Optional flip for inward-facing characters
        midpoint = num_images // 2
        if num_images >= 2:
            if (num_images % 2 == 0 and i < midpoint) or (num_images % 2 == 1 and i < midpoint):
                img = cv2.flip(img, 1)

        # Resize and position
        resized = resize_to_diagonal(img, diag_target)
        new_h, new_w = resized.shape[:2]

        x_center = positions_x[i]
        x_offset = max(0, min(WIDTH - new_w, x_center - new_w // 2))
        y_offset = (HEIGHT - new_h) // 2

        if x_offset + new_w > WIDTH or y_offset + new_h > HEIGHT:
            print(f"⚠️ Skipping character {i} due to canvas overflow")
            continue

        # Composite with alpha
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                alpha * resized[:, :, c] +
                (1 - alpha) * canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3] = np.maximum(
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3],
            resized[:, :, 3]
        )

    # Convert to canny input
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(rgb_canvas, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_pil = Image.fromarray(canny).convert("RGB")

    # Prompt construction
    layout_hint = ""
    if layout_order and len(layout_order) > 1:
        layout_hint = ", ".join([f"{name} is on the {pos}" for name, pos in zip(layout_order, position_words(len(layout_order)))]) + ". "

    appearance_hint, negative_background_hint = describe_character_appearance(used_characters)
    full_prompt = f"{layout_hint}{appearance_hint} {prompt}"
    print(full_prompt)

    # Run inference
    images = pipeline(
        prompt=full_prompt,
        negative_prompt=negative_background_hint + NEGATIVE_PROMPT,
        image=canny_pil,
        num_inference_steps=NUM_INF_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        num_images_per_prompt=NUM_IMAGES
    ).images
    print("Prompt: ", full_prompt)
    print("Negative Prompt: ", negative_background_hint + NEGATIVE_PROMPT)

    if NUM_IMAGES == 1:
        path = output_path / f"{output_file}.png"
        images[0].save(path)
        print(f" ✅Saved cover image to {path}")
    else: 
        for i, img in enumerate(images):
            path = output_path / f"{output_file}_{i:02d}.png"
            img.save(path)
            print(f" ✅Saved cover image to {path}")
