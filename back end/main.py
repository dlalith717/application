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