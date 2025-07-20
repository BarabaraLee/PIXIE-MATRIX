# Path to your book configuration file
# Usage: 
# run "SKETCH_DIR={path_to_sketches} make run" if you have sketches
# or just "make run" if you don't have sketches

# === Configurable Variables ===
CONFIG=src/config/book_config.json
PYTHON=python3

# === Application Execution ===

# Run the storybook generator with optional sketch directory
run:
	$(PYTHON) src/main.py --config $(CONFIG) --sketch_dir "$(SKETCH_DIR)"

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