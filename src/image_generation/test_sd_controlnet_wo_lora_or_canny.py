from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load models
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)

pipe.enable_xformers_memory_efficient_attention()

# Prompt
prompt = "a watercolor illustration of a bear in a forest"

# Use an existing RGB photo instead of your composed control image for test
from PIL import Image
image = Image.open("base_sd_test.png").convert("RGB").resize((768, 512))

# Get canny edge image
canny = CannyDetector()
control_image = canny(image)

# Run inference
output = pipe(
    prompt=prompt,
    image=control_image,
    guidance_scale=9,
    num_inference_steps=30
)

output.images[0].save("test_controlnet_output.png")
print("✅ Saved test_controlnet_output.png")
