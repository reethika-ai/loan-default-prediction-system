import sys
import os
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = "artifacts/model.pkl"
            self.preprocessor_path = "artifacts/preprocessor.pkl"
        except Exception as e:
            raise CustomException(e, sys)
    def load_objects(self):
        try:
            with open(self.model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(self.preprocessor_path, 'rb') as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)
            return model, preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def predict(self, input_data):
        try:
            model, preprocessor = self.load_objects()
            logging.info("transforming input data")
            data_scaled = preprocessor.transform(input_data)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)