import os
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_folder(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
        logger.info(f"Removed existing {dst}")

    logger.info(f"Path of archive folder: {src}")
    if os.path.exists(src):
        shutil.copytree(src, dst)
        logger.info(f"Restored {src} -> {dst}")
    else:
        logger.info(f"WARNING: {src} does not exist in archive, nothing restored.")

def main():
    """Restore intermediate results from an archived folder.
    Usage:
    python3 libs/restore_intermediate.py -- toddlerbook_chatgpt_draft_code/old_intermediate_results_2_2025-07-23
    """
    parser = argparse.ArgumentParser(description="Restore intermediate data from an archive folder.")
    parser.add_argument("archive_folder", help="Path to archive folder, e.g., toddlerbook_chatgpt_draft_code/old_intermediate_results_1_2025-07-22")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logger.info(f"Project root: {PROJECT_ROOT}")
    # --- CHANGED HERE: page_images parallel to src ---
    PAGE_IMAGES = os.path.join(PROJECT_ROOT, "page_images")
    INTERMEDIATE_RESULTS = os.path.join(PROJECT_ROOT, "src", "intermediate_results")

    ARCHIVE_PAGE_IMAGES = os.path.join(PROJECT_ROOT, args.archive_folder, "page_images")
    ARCHIVE_INTERMEDIATE_RESULTS = os.path.join(PROJECT_ROOT, args.archive_folder, "intermediate_results")


    restore_folder(ARCHIVE_PAGE_IMAGES, PAGE_IMAGES)
    restore_folder(ARCHIVE_INTERMEDIATE_RESULTS, INTERMEDIATE_RESULTS)

    logger.info("Restore complete.")

if __name__ == "__main__":
    main()
