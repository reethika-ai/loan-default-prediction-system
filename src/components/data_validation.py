import os
import sys 
import pandas as pd
import yaml
from src.logger import logging
from src.exception import CustomException
class DataValidation:
    def __init__(self,config_path="config.yaml"):
        try:
            with open(config_path,"r") as file:
                self.config=yaml.safe_load(file)
                self.required_columns=self.config["data_validation"]["required_columns"]
        except Exception as e:
            raise CustomException(e,sys)
    def validate_dataset(self,file_path):
        try:
            logging.info(f"Starting data validation for{file_path}")
            if not os.path.exists(file_path):
                raise Exception(f"Dataset file not found at{file_path}")
            df=pd.read_csv(file_path)
            logging.info(f"Dataset loaded with shape{df.shape}")
            for col in self.required_columns:
                if col not in df.columns:
                    raise Exception(f"Missing required colum:{col}")
            logging.info("All required columns are present")
            missing_count=df.isnull().sum().sum()
            if missing_count > 0:
                logging.warning(f"Dataset contains {missing_count}missing values")
            else:
                logging.info("No missing values detected")
                logging.info("Data validation completed successfully")
                return True
        except Exception as e:
            raise CustomException(e,sys)