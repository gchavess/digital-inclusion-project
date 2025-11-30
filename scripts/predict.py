# Usa o modelo treinado para prever novos valores

import pandas as pd
import pickle
import json

# Carregar modelo
with open("models/modelo_final.pkl", "rb") as f:
    model = pickle.load(f)

# Carregar lista de features usadas no treinamento
with open("models/features.json", "r") as f:
    features = json.load(f)

# Exemplo: prever para um país
novo_pais = {
    "Latitude": -15.78,
    "Longitude": -47.93,
    "GDP_per_capita": 13000,
    "E_Commerce_Penetration (%)": 55,
    "Device_Penetration (%)": 78,
    "Government_Digital_Policy_Index (%)": 62,
    "Data_Privacy_Regulation_Strength (%)": 70,
    "Urbanization_Rate (%)": 87,
    "Education_Index": 0.73,
    "Mobile_Data_Price_USD": 2.1,
    # ... COMPLETE TODAS AS FEATURES AQUI
}

# Criar um DF com todas as colunas
df = pd.DataFrame([novo_pais])

# Garantir que as colunas estejam na ordem correta
df = df.reindex(columns=features)

# Prever
pred = model.predict(df)[0]

print(f"Previsão: {pred}")
