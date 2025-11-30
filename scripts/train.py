# Cria o modelo (aprende com os dados)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import json

DATA_PATH = Path("data/internet_adoption_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "Internet_Penetration (%)"
features = df.select_dtypes(include="number").columns.tolist()
features.remove(target)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Salvar modelo com compressão
dump(pipeline, MODEL_DIR / "modelo_final.joblib", compress=3)

# Salvar features
with open(MODEL_DIR / "features.json", "w") as f:
    json.dump(features, f)

print("Modelo treinado e salvo em models/modelo_final.joblib")
print("Features salvas em models/features.json")
