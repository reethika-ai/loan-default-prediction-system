import sys
import os
import json
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from src.logger import logging
from src.exception import CustomException
class ModelEvaluation:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.metrics_to_track = self.config['model_evaluation']['metrics_to_track']
                self.metrics_artifact = self.config['model_evaluation']['metrics_artifact']
            os.makedirs(os.path.dirname(self.metrics_artifact), exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)
    def evaluate_model(self, y_true,y_pred):
        try:
            results={}
            if "accuracy" in self.metrics_to_track:
                results["accuracy"]=accuracy_score(y_true,y_pred)
            if "precision" in self.metrics_to_track:
                results["precision"]=precision_score(y_true,y_pred,pos_label='Y')
            if "recall" in self.metrics_to_track:
                results["recall"]=recall_score(y_true,y_pred,pos_label='Y')
            if "f1_score" in self.metrics_to_track:
                results["f1_score"]=f1_score(y_true,y_pred,pos_label='Y')
            cm=confusion_matrix(y_true,y_pred)
            logging.info(f"Evaluation results: {results}")
            logging.info(f"Confusion Matrix:\n{cm}")
            with open(self.metrics_artifact, 'w') as file:
                json.dump(results, file, indent=4)
            return results,cm
        except Exception as e:
            raise CustomException(e, sys)