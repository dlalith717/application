import sqlite3

def create_connection():
    conn=sqlite3.connect("patients.db")
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        gender TEXT,
        symptoms TEXT,
        bp INTEGER,
        heart_rate INTEGER,
        temperature REAL,
        conditions TEXT,
        risk TEXT,
        department TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from model import predict_risk
from database import create_table, create_connection
import shutil

app = FastAPI()

create_table()

class PatientInput(BaseModel):
    age: int
    gender: str
    symptoms: str
    blood_pressure: int
    heart_rate: int
    temperature: float
    conditions: str


def recommend_department(risk, symptoms):
    if risk == "High":
        return "Emergency"
    if "chest" in symptoms.lower():
        return "Cardiology"
    if "headache" in symptoms.lower():
        return "Neurology"
    return "General Medicine"


@app.post("/predict")
def predict(patient: PatientInput):

    risk, confidence, explanation = predict_risk(
        patient.age,
        patient.blood_pressure,
        patient.heart_rate,
        patient.temperature
    )

    department = recommend_department(risk, patient.symptoms)

    # Save to DB
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients 
    (age, gender, symptoms, bp, heart_rate, temperature, conditions, risk, department, confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient.age,
        patient.gender,
        patient.symptoms,
        patient.blood_pressure,
        patient.heart_rate,
        patient.temperature,
        patient.conditions,
        risk,
        department,
        confidence
    ))

    conn.commit()
    conn.close()

    return {
        "Risk_Level": risk,
        "Recommended_Department": department,
        "Confidence_Score": round(confidence, 2),
        "Explainability": explanation
    }


@app.post("/upload-ehr")
def upload_file(file: UploadFile = File(...)):
    with open("uploaded_ehr.txt", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "EHR uploaded successfully"}

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

import pandas as pd
import numpy as np

def generate_synthetic_data(n=500):
    data = []

    for i in range(n):
        age = np.random.randint(1, 90)
        bp = np.random.randint(90, 180)
        hr = np.random.randint(60, 140)
        temp = round(np.random.uniform(97, 103), 1)

        risk = "Low"
        if bp > 160 or hr > 120 or temp > 101:
            risk = "High"
        elif bp > 140 or hr > 100:
            risk = "Medium"

        data.append([age, bp, hr, temp, risk])

    df = pd.DataFrame(data, columns=[
        "Age", "BloodPressure", "HeartRate", "Temperature", "Risk_Level"
    ])

    df.to_csv("synthetic_data.csv", index=False)
    return df