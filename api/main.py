from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle

# загрузка модели
model=pickle.load(open("model\predictor_model1.pkl",'rb'))

app=FastAPI()

class Input(BaseModel):
    bilirubin:float
    Neutrophils:int
    Amylase:float
    Duration:int
    Lymphocytes:int

@app.get("/")
def read_root():
    return {"msg":"Predictor"}

@app.post("/predict")
def predict_price(input:Input):
    data = input.dict()
    data_in = [[data['bilirubin'], data['Neutrophils'], data['Amylase'], data['Duration'],data['Lymphocytes']]]

    prediction = model.predict(data_in)
    return {
        'prediction': prediction[0]
        }

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
