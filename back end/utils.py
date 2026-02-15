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