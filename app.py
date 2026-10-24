# Step 1 — activate environment(On one terminal)
#     cd "C:\Users\Suyash\Pictures\FastAPI_learn"
#     .\fast_env\Scripts\activate

#      Step i) — Start FastAPI
#          python -m uvicorn app:app --reload (Type this)
#          Keep this terminal running.
#          You should have:
#          http://127.0.0.1:8000


# Step 2 — Open another terminal
# Activate the same environment:
#       cd "C:\Users\Suyash\Pictures\FastAPI_learn"
#       .\fast_env\Scripts\activate

#    Step ii) — Start Streamlit
#          streamlit run frontend.py 
#          Then open:
#          http://localhost:8501

#Note there is one pdf MLmodel_Frontend_.... in same folder where you can find the details of the small project
#(In this project we are using only three files from this folder --app.py , frontend.py , model.pkl(which is generated after running fastapi_ml_model.py))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output,MODEL_VERSION,model



app = FastAPI()






@app.get('/')
def home():
    return {'message':'insurance_premium_prediction+_API'}

@app.get('/health')
def health_check():
    return {'status':'OK', 'model_version':MODEL_VERSION ,'model_loaded':model is not None }

@app.post('/predict',response_model=PredictionResponse) #PredictionResponse is pydantic model's class imported from schema.prediction_response, here we validating the  data with pydantic class 
def predict_premium(data: UserInput): #UserInput is pydantic model's class schema.puser_input , here we converting the input data into pydantic class object

    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
       # Prediction
       prediction = predict_output(user_input) #function from predict.py file 

       return JSONResponse(status_code=200,content={'Response': str(prediction)})
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))




