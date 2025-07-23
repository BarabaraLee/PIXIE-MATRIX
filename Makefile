# Path to your book configuration file
# Usage: 
# run "SKETCH_DIR={path_to_sketches} make run" if you have sketches
# or just "make run" if you don't have sketches

# === Configurable Variables ===
CONFIG=src/config/book_config.json
SKETCH_DIR=
INTERMEDIATE=src/intermediate_results/intermediate.json

# === Application Execution ===

# Run the storybook generator with optional sketch directory
.PHONY: run run-gen run-assemble

# Run the full pipeline (default: phase=gen, then assemble)
run: run-gen run-assemble

# Run steps 1+2, saving to intermediate file
run-gen:
	python3 src/main.py --phase gen --config $(CONFIG) --sketch_dir "$(SKETCH_DIR)" --intermediate_file $(INTERMEDIATE)

# Run steps 3+4+5, loading from intermediate file
run-assemble:
	python3 src/main.py --phase assemble --intermediate_file $(INTERMEDIATE)

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

# === Clean Up ===

clean:
	@echo "🧹 Removing generated files..."
	rm -f *.epub *.png *.jpg *.jpeg

# === Full End-to-End Flow ===

e2e: test clean
	@echo "✅ End-to-end validation complete."