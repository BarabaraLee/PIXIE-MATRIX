# 📚 Toddler Storybook Generator

This project generates fully illustrated EPUB storybooks for toddlers using generative AI models (OpenAI GPT, Gemma, Stable Diffusion XL, and ControlNet). You provide a theme and optional character poses, and it outputs a complete, publishable e-book. (under developement)

---

## 🚀 How to Run

### 0. Setup virtual environment
* mac (with cpu only): source env/bin/activate (virtual-environment initiation: python -m venv env)
* windows (with gpu): source win-env/Scripts/activate (V-environment initiation: ../python.exe -m venv win-env2)

Makesure python version is at least 3.13.5 before proceeds.

### 1. Install dependencies

pip install -r requirements.txt


---
## 2. Add API keys to .env file

OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_TOKEN=your_hf_token


---
## 3. Make sure you have access to the desired model
For default model to generate text, we currently use gemma-2b-it mode, which requires the following:
* Make sure to have access to it at https://huggingface.co/google/gemma-2b-it
* (optional) download the pre-trained model, run in terminal: transformers-cli download google/gemma-2b-it

Also, the LoRA model training for the characters are completed on Google Colab with GPU support. Currently I only trained for 2 characters (a cat and a bear). You can be innovative to find decent pretrained model (but this will require you to fine tune image at later image generation stage). 

---
## 4. Run the app

Opiton 1: 
* Branch of code: main
* Book generation mode: generate book only with prompt (no guidance from sketch/skeleton pictures)
* How it is realized: generate sentences (and image generation page descriptions) from title, subtitle, characters invovled and story theme. Then generate 3-4 images for each of the page descriptions. Let user pick 1 image for each page. Then automatically place the story sentencies to page images for each page (including 2 cover pages) and generate a ebook of EPub format (good for Apple book store).
* Command: make run-gen, make run-assemble
* Status: Almost done.
* ToDos: complete test coverage, clean the code.

Option 2: (Preferred approach)
* Branch of code: method3-3-stage-run-txt-run-image-run-assemble
* Book generation mode: generate book with prompt + prepared character images.
* Details: generate sentences (and image generation page descriptions) from title, subtitle, characters invovled and story theme. Let user to prepare 10-20 images (of different poses) for each character (train with LoRA). Use prompt to ask user for selecting character images and character positions in the page for all the generated page description. Then generate book pages with LoRA+ControlNet+StableDiffusion. Then automatically place the story sentencies to page images for each page (including 2 cover pages) and generate a ebook of EPub format (good for Apple book store).
* Command: make run-txt, make run-image, make run-assemble
* Status: Under development. 
* Todos: develop run-assemble (should be similar to that of Option 1), add test cases, clean code (remove the lagacy code from the main branch).

TODO: 
* move the branch method3-3-stage-run-txt-run-image-run-assemble to a separate repo. 
* Update Makefile

# How to test

## 1. Run all test
make test

