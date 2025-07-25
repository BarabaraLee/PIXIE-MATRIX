# Path to your book configuration file
# Usage: 
# run "SKETCH_DIR={path_to_sketches} make run" if you have sketches
# or just "make run" if you don't have sketches

# === Configurable Variables ===
PYTHON=python3
CONFIG=src/config/book_config.json
SKETCH_DIR=
INTERMEDIATE=src/intermediate_results/intermediate.json

# === Application Execution (New4-stage approach support)===

# Run the storybook generator with optional sketch directory
.PHONY: run run-characters run-story run-images run-assemble run-full-pipeline

run-characters:
	@echo " Stage 1: Character Generation"
	$(PYTHON) src/character_generation/character_agent.py 

run-story:
	@echo " Stage 2: Story Generation"
	$(PYTHON) src/story_generation/story_agent.py 

run-images:
	@echo " Stage 3: Image Generation"
	$(PYTHON) src/image_generation/character_aware_gen.py 

run-assemble:
	@echo "Stage 4: Book Assembly"
	$(PYTHON) src/assembly/character_integration.py

# === Application Execution (Lagacy 2-stage approach support)===
run-gen:
	$(PYTHON) src/main.py --phase gen --config $(CONFIG) --sketch_dir "$(SKETCH_DIR)" --intermediate_file $(INTERMEDIATE)

# run-assemble:
# 	$(PYTHON) src/main.py --phase assemble --intermediate_file $(INTERMEDIATE)

# === Unit Tests ===

test:
	@echo "🔍 Running all unit tests..."
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test-epub:
	@echo "📘 Testing EPUB assembler..."
	$(PYTHON) -m unittest tests/test_epub_assembler.py

test-story:
	@echo "📖 Testing story generation..."
	$(PYTHON) -m unittest tests/test_story_generator.py

test-cover:
	@echo "🖼️  Testing cover page generation..."
	$(PYTHON) -m unittest tests/test_cover_creator.py

# === Clean Up ===

clean:
	@echo "🧹 Removing generated files..."
	rm -f *.epub *.png *.jpg *.jpeg

# === Full End-to-End Flow ===

e2e: test clean
	@echo "✅ End-to-end validation complete."