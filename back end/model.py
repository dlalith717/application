import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from utils import generate_synthetic_data

MODEL_PATH = "triage_model.pkl"

def train_model():
    df = generate_synthetic_data()

    X = df[["Age", "BloodPressure", "HeartRate", "Temperature"]]
    y = df["Risk_Level"]

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = train_model()
    return model

def predict_risk(age, bp, hr, temp):
    model = load_model()

    data = pd.DataFrame([[age, bp, hr, temp]],
                        columns=["Age", "BloodPressure", "HeartRate", "Temperature"])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data).max()

    # Explainability (feature importance)
    importance = model.feature_importances_
    features = ["Age", "BloodPressure", "HeartRate", "Temperature"]

    explanation = dict(zip(features, importance.round(3)))

    return prediction, float(probability), explanation