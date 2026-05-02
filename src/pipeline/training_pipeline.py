import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import pickle
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
        trainer = ModelTrainer()
        best_model_name, best_model_score = trainer.initiate_model_training(X_train_transformed, y_train, X_test_transformed, y_test)
        print(f"Best model: {best_model_name}")
        print(f"Best model score: {best_model_score}")
        with open("artifacts/model.pkl", 'rb') as file:
           best_model = pickle.load(file)
        test_df=pd.read_csv(test_path)
        y_test=test_df["Loan_Status"]
        with open("artifacts/preprocessor.pkl", 'rb') as file:
            preprocessor = pickle.load(file)
        X_test=preprocessor.transform(test_df.drop(columns=["Loan_Status"]))
        evaluator=ModelEvaluation()
        results, cm = evaluator.evaluate_model(y_test, best_model.predict(X_test))
        print("Evaluation results:", results)
        print("Confusion Matrix:\n", cm)
    except Exception as e:
        raise CustomException(e, sys)
