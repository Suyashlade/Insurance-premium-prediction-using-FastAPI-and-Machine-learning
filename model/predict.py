import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f: #Read Binary
    model = pickle.load(f)

MODEL_VERSION='1.0.0'

class_labels=model.classes_.tolist()  #['high','low','medium'] this list of output classes of model is created


def predict_output(user_input: dict):

    df=pd.DataFrame([user_input])

    predicted_class =model.predict(df)[0]

    probabilities=model.predict_proba(df)[0]
    confidence=max(probabilities)

    class_probs=dict(zip(class_labels,map(lambda p: round(p,4),probabilities)))


    return {
        "predicted category":predicted_class,
        "confidence":round(confidence,4),
        "class_probablities":class_probs
    }
