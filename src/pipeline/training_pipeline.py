import sys
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(
            source_file_path="data/loan_data.csv"
        )
        print("Train file:", train_path)
        print("Test file:", test_path)
    except Exception as e:
        raise CustomException(e, sys)