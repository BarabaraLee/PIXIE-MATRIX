# 📚 Toddler Storybook Generator

This project generates fully illustrated EPUB storybooks for toddlers using generative AI models (OpenAI GPT, Groq Gemma, Stable Diffusion XL, and ControlNet). You provide a theme and optional layout sketches, and it outputs a complete, publishable e-book.

---

## 🗂️ Source Code Structure
src
├── config
│   └── book_config.json
├── main.py
└── modules
    ├── __init__.py
    ├── cover_creator.py
    ├── epub_assembler.py
    ├── illustration_generator.py
    ├── story_generator.py
    └── text_placer.py


---

## 🚀 How to Run

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


---
## 3. Run the app

Opiton 1: 
* Book generation mode: generate book only with prompt (no guidance from sketch/skeleton pictures)
* Command: make run

Option 2: 
* Book generation mode: provide a folder of layout sketches (the sketches will help to guide backgound contests of pictures and help for keeping consistency between pages).
* Command: make run SKETCH_DIR={sketch_folder_path}


# How to test

## 1. Run all test
make test

## Run individual test modules:
make test-epub           # Test EPUB assembler logic
make test-story          # Test story generator (OpenAI + Groq)

