import os
import shutil
import yaml
from utils.logger import logger
from utils.exception import CustomException
import sys

def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def copy_raw_data(config: dict):
    try:
        source = config["data_source"]["kaggle_download_path"]
        destination = config["data_source"]["local_artifact_path"]
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)

        logger.info(f"Dataset copied from {source} to {destination}")
        print(f"Dataset copied to: {destination}")

    except Exception as e:
        raise CustomException(str(e), sys)

if __name__ == "__main__":
    config = load_config()
    copy_raw_data(config)
