HEIGHT = 768 # 1536 
WIDTH = 512 # 1024

CHARACTOR_HEIGHT = 256
NUM_INF_STEPS = 30
NUM_IMAGES = 2
NEGATIVE_PROMPT = (
    "blurry, bad anatomy, malformed limbs, missing or extra fingers, extra eyes, twisted faces, poor proportions, unrealistic, creepy, "
    "nsfw, violent, animal-human hybrid, animals wearing clothes, glitches, watermark, text, cluttered, chaotic, unnatural lighting, "
    "wrong object count, disorganized, poor composition, monochrome, weird colors, letters, nudity"
)
GUIDANCE_SCALE = 9
IMAGE_STYLE = "Watercolor"
MAX_NEW_OUTPUT_TOKENS = 4096  # Maximum tokens for model output

DO_SAMPLE=False # Deterministic Story Generation
NUM_VERSIONS=1