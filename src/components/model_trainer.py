import sys
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
class ModelTrainer:
    def __init__(self,config_path="config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.model_path = self.config['model_trainer']['model_path']
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models={
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier()
            }
            model_report={}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred=model.predict(X_test)
                acc=accuracy_score(y_test, y_pred)
                model_report[name]=acc
                logging.info(f"{name} accuracy: {acc}")
                return model_report,models
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info(" model training started")
            model_report,models=self.initiate_model_trainer(X_train, y_train, X_test, y_test)
            best_model_name=max(model_report, key=model_report.get)
            best_model_score=model_report[best_model_name]
            best_model=models[best_model_name]
            logging.info(f"best model:{best_model_name}")
            logging.info(f"best model score:{best_model_score}")
            with open(self.model_path, 'wb') as file:
                pickle.dump(best_model, file)
            logging.info(f"best model saved at {self.model_path}")
            return best_model_name,best_model_score
        except Exception as e:
            raise CustomException(e, sys)