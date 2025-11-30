#$ python3 scripts/predict.py --model RandomForest --input data/internet_adoption_clean_final.csv        

# scripts/predict.py
import pandas as pd
import joblib
import argparse
import os

MODEL_DIR = "models"

def main(model_name, input_file):
    # Carregar modelo
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"Modelo {model_name} não encontrado!")
        return
    
    model = joblib.load(model_path)
    
    # Carregar dados de entrada
    df_input = pd.read_csv(input_file)
    
    # Renomear colunas para coincidir com treino
    df_input.columns = [c.strip().replace(' ', '_')
                        .replace('(', '').replace(')', '')
                        .replace('%', 'pct') for c in df_input.columns]
    
    # Remover coluna alvo se existir
    TARGET = 'Internet_Penetration_pct'
    if TARGET in df_input.columns:
        df_input = df_input.drop(columns=[TARGET])
    
    # Selecionar apenas colunas numéricas
    X_input = df_input.select_dtypes(include='number')
    
    # Gerar previsões
    predictions = model.predict(X_input)
    
    df_input['prediction'] = predictions
    print("\nPrevisões:\n", df_input.head(10))
    
    # Salvar previsões em CSV
    output_file = os.path.join(MODEL_DIR, f"predictions_{model_name}.csv")
    df_input.to_csv(output_file, index=False)
    print(f"\nPrevisões salvas em {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Nome do modelo: LinearRegression, RandomForest, XGBoost")
    parser.add_argument("--input", type=str, required=True,
                        help="Arquivo CSV de entrada")
    args = parser.parse_args()
    
    main(args.model, args.input)
