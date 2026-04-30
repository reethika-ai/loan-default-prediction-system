import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.exception import CustomException
class DataIngestion:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            self.raw_data_path = self.config["data_ingestion"]["raw_data_path"]
            self.train_data_path = self.config["data_ingestion"]["train_data_path"]
            self.test_data_path = self.config["data_ingestion"]["test_data_path"]
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_ingestion(self, source_file_path):
        try:
            logging.info("Starting data ingestion")
            df = pd.read_csv(source_file_path)
            logging.info(f"Dataset read successfully with shape {df.shape}")
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.raw_data_path}")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)
            logging.info(f"Train data saved at {self.train_data_path}")
            logging.info(f"Test data saved at {self.test_data_path}")
            return (
                self.train_data_path,
                self.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)