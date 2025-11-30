# Cria o modelo (aprende com os dados)

import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Caminhos
DATA_PATH = Path("data/internet_adoption_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Carregar dados
df = pd.read_csv(DATA_PATH)

# Exemplo: prever Internet_Penetration (%)
target = "Internet_Penetration (%)"

# Features numéricas (todas menos a target e colunas não numéricas)
features = df.select_dtypes(include="number").columns.tolist()
if target in features:
    features.remove(target)

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: scaler + modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

# Treinar
pipeline.fit(X_train, y_train)

# Salvar modelo
with open(MODEL_DIR / "modelo_final.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Salvar features usadas
with open(MODEL_DIR / "features.json", "w") as f:
    json.dump(features, f)

print("Modelo treinado e salvo em models/modelo_final.pkl")
print("Features salvas em models/features.json")
