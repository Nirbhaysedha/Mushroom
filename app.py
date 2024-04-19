from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd

class PredictionInput(BaseModel):
    cap_diameter: int
    cap_shape: int
    gill_attachment: int
    gill_color: int
    stem_height: float
    stem_width: int
    stem_color: int
    season: float


app=FastAPI()
model_path='models/model.joblib'
model=load(model_path)

@app.get('/')
def home():
    return "hey we are on successfull!"

@app.post('/predict')
def predict(input_data: PredictionInput):
        features={
    "cap-diameter":input_data.cap_diameter,
    "cap-shape":input_data.cap_shape,
    "gill-attachment":input_data.gill_attachment,
    "gill-color":input_data.gill_color,
    "stem-height":input_data.stem_height,
    "stem-width":input_data.stem_width,
    "stem-color":input_data.stem_color,
    "season":input_data.season
        }
        feat=pd.DataFrame(features,index=[0])
        prediction = model.predict(feat)[0].item()
        return f"predictions : {prediction}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



