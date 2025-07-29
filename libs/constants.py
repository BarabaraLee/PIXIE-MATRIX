HEIGHT = 512 # 1536 
WIDTH = 768 # 1024

NUM_INF_STEPS = 30
NUM_IMAGES = 4
NEGATIVE_PROMPT = "blurry, bad anatomy, malformed limbs, extra limbs, extra eyes, poor proportions, glitch, watermark, text, cluttered, chaotic, nsfw, violent, animal-human hybrid, clothing, nudity, human features"

# (
# "blurry, bad anatomy, malformed limbs, missing or extra fingers, extra eyes,"
# "poor proportions, unrealistic, nsfw, violent, animal-human hybrid, animals in clothes, "
# "glitch, watermark, text, cluttered, chaotic, unnatural light, wrong object count, disorganized, "
# "bad composition, monochrome, weird colors, letters, nudity, human, human limbs, clothing on animal"
# ) #twisted face, creepy
GUIDANCE_SCALE = 9.5
IMAGE_STYLE = "Watercolor"
MAX_NEW_OUTPUT_TOKENS = 4096  # Maximum tokens for model output

DO_SAMPLE=False # Deterministic Story Generation
NUM_VERSIONS=1
CONTROLNET_CONDITIONING_SCALE=1.0