import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_next_backup_folder(base_dir, prefix="old_intermediate_results"):
    idx = 1
    date_str = datetime.now().strftime("%Y-%m-%d")
    while True:
        folder_name = f"{prefix}_{idx}_{date_str}"
        full_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(full_path):
            return full_path
        idx += 1

def copy_folder(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst)
        logger.info(f"Copied {src} -> {dst}")
    else:
        logger.info(f"WARNING: {src} does not exist and was not copied.")

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DRAFT_DIR = os.path.join(PROJECT_ROOT, "toddlerbook_chatgpt_draft_code")
    # --- CHANGED HERE: page_images parallel to src ---
    PAGE_IMAGES = os.path.join(PROJECT_ROOT, "page_images")
    INTERMEDIATE_RESULTS = os.path.join(PROJECT_ROOT, "src", "intermediate_results")

    os.makedirs(DRAFT_DIR, exist_ok=True)
    backup_folder = find_next_backup_folder(DRAFT_DIR)
    os.makedirs(backup_folder, exist_ok=True)

    copy_folder(PAGE_IMAGES, os.path.join(backup_folder, "page_images"))
    copy_folder(INTERMEDIATE_RESULTS, os.path.join(backup_folder, "intermediate_results"))

    logger.info(f"All intermediate results archived to {backup_folder}")

if __name__ == "__main__":
    main()
