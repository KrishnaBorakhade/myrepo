from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load dataset (Ensure dataset.txt is in the same folder)
DATASET_PATH = os.getenv("DATASET_PATH", "dataset.txt")  # Use env variable or default file
data = pd.read_csv(DATASET_PATH)

# Request model
class SoilFertilityRequest(BaseModel):
    NO3: float
    NH4: float
    P: float
    K: float
    SO4: float
    B: float
    OrganicMatter: float
    pH: float
    Zn: float
    Cu: float
    Fe: float
    Ca: float
    Mg: float
    Na: float

@app.post("/predict_soil_fertility")
def predict_soil_fertility(request: SoilFertilityRequest):
    user_input = [
        request.NO3, request.NH4, request.P, request.K, request.SO4, request.B,
        request.OrganicMatter, request.pH, request.Zn, request.Cu, request.Fe,
        request.Ca, request.Mg, request.Na
    ]
    
    # Prepare dataset for model
    X, Y = data[data.columns[1:]], data['Vegetation Cover']
    df1 = pd.DataFrame([user_input], columns=['NO3', 'NH4', 'P', 'K', 'SO4', 'B', 
                                              'Organic Matter', 'pH', 'Zn', 'Cu', 'Fe', 'Ca', 'Mg', 'Na'])
    df = pd.concat([X, df1], ignore_index=True)
    
    # Normalize data
    scaler = MinMaxScaler()
    X, Y = scaler.fit_transform(X.values), scaler.fit_transform(Y.values.reshape(-1, 1))
    
    # Prepare input sample
    l1 = [X[-1]]  
    X = X[:-1]  

    # Train RandomForestRegressor
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=43)
    forestRegressor = RandomForestRegressor(criterion='squared_error', max_depth=8, n_estimators=10, random_state=0)
    forestRegressor.fit(X_train, Y_train)
    prediction = forestRegressor.predict(l1)[0]

    # Interpret results
    if prediction < 0.9:
        deficiencies = []
        if request.NO3 < 12.75:
            deficiencies.append("NO3")
        if request.P < 47:
            deficiencies.append("P")
        if request.Zn < 0.6:
            deficiencies.append("Zn")
        if request.K < 15:
            deficiencies.append("K")
        if request.OrganicMatter < 0.28:
            deficiencies.append("Organic Matter")
        if request.Fe < 1:
            deficiencies.append("Fe")

        return {
            "fertility_status": "low",
            "suggestion": f"Your soil is less fertile. Consider increasing: {', '.join(deficiencies)}",
            "fertility_percentage": round(prediction * 100)
        }
    
    return {
        "fertility_status": "high",
        "suggestion": "Your soil is highly fertile.",
        "fertility_percentage": round(prediction * 100)
    }

# Production-ready server setup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
