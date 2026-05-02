import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
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
        validator=DataValidation()
        validator.validate_dataset(train_path)
        validator.validate_dataset(test_path)
        print("Data ingestion and validation completed successfully")
        transformer = DataTransformation()
        X_train_transformed, y_train, X_test_transformed, y_test = transformer.initiate_data_transformation(train_path, test_path)
        print("Data transformation completed successfully")
    except Exception as e:
        raise CustomException(e, sys)
