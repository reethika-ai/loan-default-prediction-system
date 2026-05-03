import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline
data=pd.DataFrame(
    {
        "Gender": ["Male"],
        "Married": ["Yes"],
        "Dependents": [0],
        "ApplicantIncome": [5000],
        "LoanAmount": [200],
        "Credit_History": [1]
    })
pipeline=PredictPipeline()
result=pipeline.predict(data)
print("prediction is ",result)