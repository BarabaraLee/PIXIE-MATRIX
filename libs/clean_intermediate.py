import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_folder(folder_path):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Deleted folder: {item_path}")
            else:
                os.remove(item_path)
                logger.info(f"Deleted file: {item_path}")
        logger.info(f"Cleaned: {folder_path}")
    else:
        logger.info(f"WARNING: {folder_path} does not exist. Nothing to clean.")

def main():
    # Find the script location and resolve project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PAGE_IMAGES = os.path.join(PROJECT_ROOT, "page_images")
    INTERMEDIATE_RESULTS = os.path.join(PROJECT_ROOT, "src", "intermediate_results")

    logger.info("This will delete ALL files and folders inside:")
    logger.info(f" - {PAGE_IMAGES}")
    logger.info(f" - {INTERMEDIATE_RESULTS}")
    resp = input("Continue? (y/n): ")
    if resp.lower() != "y":
        logger.info("Aborted.")
        return

    clean_folder(PAGE_IMAGES)
    clean_folder(INTERMEDIATE_RESULTS)
    logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()
