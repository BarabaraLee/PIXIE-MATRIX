import os
import sys
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# --------- Command-line input path ---------
if len(sys.argv) != 2:
    print("Usage: python3 cartoonify_batch.py /path/to/input_dir")
    sys.exit(1)

input_dir = sys.argv[1]
if not os.path.isdir(input_dir):
    print(f"Error: Input path '{input_dir}' is not a directory.")
    sys.exit(1)

output_dir = os.path.join(input_dir, "cartoon")
os.makedirs(output_dir, exist_ok=True)

# --------- Load Stable Diffusion pipeline ---------
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_safetensors=True
).to("cpu")

prompt = "2D water color illustration of a cute animal, colorful, outlined, soft lighting, for children's picture book"
negative_prompt = "photo, realistic, 3D, dark, distorted, blurry"

supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --------- Helper to load + resize image ---------
def prepare_input_image(image_path):
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    resized_image = original_image.resize((512, 512))
    return resized_image, original_size

# --------- Batch Process ---------
for filename in os.listdir(input_dir):
    if filename.lower().endswith(supported_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cartoon.png")

        print(f"Processing: {filename}")
        resized_image, original_size = prepare_input_image(input_path)

        output = pipe(
            prompt=prompt,
            image=resized_image,
            strength=0.6,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=50
        )

        cartoon_image = output.images[0].resize(original_size, resample=Image.BICUBIC)
        cartoon_image.save(output_path)
        print(f"Saved: {output_path}")

print("✅ All images cartoonified and saved.")
