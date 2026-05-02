import sys
import os
import pandas as pd
import pickle
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from src.exception import CustomException
class DataTransformation:
    def __init__(self,config_path="config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.preprocessor_path = self.config['data_transformation']['preprocessor_path']
                self.target_column= self.config['data_transformation']['target_column']
                self.numerical_columns = self.config['data_transformation']['numerical_columns']
                self.categorical_columns = self.config['data_transformation']['categorical_columns']
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)
    def get_preprocessor(self):
        try:
            logging.info("Creating preprocessor object")
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("numerical", numerical_pipeline, self.numerical_columns),
                    ("categorical", categorical_pipeline, self.categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Splitting input and target")
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]
            preprocessor = self.get_preprocessor()
            logging.info("fitting preprocessor on training data and transforming both training and testing data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Saving preprocessor object")
            with open(self.preprocessor_path, 'wb') as file:
                pickle.dump(preprocessor, file)
            return X_train_transformed, y_train, X_test_transformed, y_test
        except Exception as e:
            raise CustomException(e, sys)
            