from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import PredictPipeline
app = FastAPI()
class LoanData(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    ApplicantIncome: int
    LoanAmount: int
    Credit_History: int
@app.get("/")
def home():
    return {"message": "loan prediction API is running"}
@app.post("/predict")
def predict(data: LoanData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        pipeline= PredictPipeline()
        result= pipeline.predict(input_df)
        print("raw result:",result)
        print("type",type(result[0]))
        pred = result[0]
        if pred in ["Y", "N"]:
            pred = 1 if pred == "Y" else 0

        return {
            "prediction": int(pred),
            "status": "Approved" if pred == 1 else "Rejected"
        }
    except Exception as e:
        return {"error": str(e)}