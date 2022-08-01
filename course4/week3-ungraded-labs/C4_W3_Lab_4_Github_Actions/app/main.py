import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

# This is a test comment 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Predicting Wine Class with batching")

# Open classifier in global scope
with open("models/wine-95-fixed.pkl", "rb") as file:
    clf = pickle.load(file)


class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}

def test_pipeline_and_scaler():
    # Check if clf is an instance of sklearn.pipeline.Pipeline 
    isPipeline = isinstance(clf, Pipeline)
    assert isPipeline
    
    if isPipeline:
        # Check if first step of pipeline is an instance of 
        # sklearn.preprocessing.StandardScaler
        firstStep = [v for v in clf.named_steps.values()][0]
        assert isinstance(firstStep, StandardScaler)