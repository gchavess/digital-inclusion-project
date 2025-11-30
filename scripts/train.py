# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Caminho dos dados
DATA_PATH = "data/internet_adoption_clean_final.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Carregar dados
df = pd.read_csv(DATA_PATH)

# Renomear colunas para facilitar uso
df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct') for c in df.columns]

# Definir alvo
TARGET = 'Internet_Penetration_pct'

# Selecionar apenas colunas numéricas como features
X = df.select_dtypes(include='number').drop(columns=[TARGET])
y = df[TARGET]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar modelos
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
}

# Treinar, avaliar e salvar resultados
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    
    # Salvar cada modelo
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

# Mostrar tabela de resultados
results_df = pd.DataFrame(results)
print("\nResultados dos modelos:\n", results_df)

# Salvar resultados em CSV
results_df.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)
