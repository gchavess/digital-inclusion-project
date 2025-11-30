# Digital Inclusion Project
Alunos: Gustavo Chaves e Michael Varaldo

## 📄 Descrição do Projeto

Este projeto tem como objetivo analisar a adoção da internet e indicadores digitais em diversos países, construindo um modelo preditivo de **Penetração de Internet (%)** a partir de variáveis socioeconômicas, de infraestrutura e inclusão digital. O foco é aplicar o ciclo completo de Ciência de Dados: coleta, tratamento, análise exploratória, modelagem e deploy do modelo.

---

## 🎯 Problema de Negócio

Nosso projeto se insere no contexto da **inclusão digital global**, onde a penetração de internet ainda apresenta disparidades significativas entre países.

**Pergunta de negócio:**  
Quais fatores socioeconômicos e tecnológicos têm maior impacto na penetração da internet em um país?

**Objetivo do modelo:**  
Construir um modelo preditivo capaz de estimar a penetração de internet (%) em países a partir de indicadores digitais, socioeconômicos e de infraestrutura.

---

## 🛠️ Pipeline de Dados

1. **Origem dos Dados**

   - Dataset: `global-internet-adoption-trends.csv` (Kaggle).
   - Contém 28 colunas com indicadores como penetração de internet, velocidade de banda, custo de internet, educação digital, entre outros.

2. **Ingestão**

   - Os dados foram carregados e salvos no formato limpo `internet_adoption_clean_final.csv`.

3. **Limpeza e Transformação**

   - Remoção de duplicatas e espaços nos nomes das colunas.
   - Preenchimento de valores ausentes: medianas para numéricos e moda para categóricos.
   - Criação de features adicionais:
     - `Total_Speed_Index`: combinação de banda fixa e móvel.
     - `Digital_Inclusion_Index`: ponderação de penetração de internet, alfabetização digital e penetração de dispositivos.
     - `Relative_Internet_Cost`: custo de acesso relativo ao PIB per capita.
     - `Log_GDP` e `Log_Internet_Cost`: transformação logarítmica.
     - `Urban_Rural_Ratio`: razão urbano/rural.
   - One-hot encoding de `5G_Rollout_Status`.

4. **Análise Exploratória (EDA)**

   - Histogramas, boxplots e matrizes de correlação para identificar padrões, outliers e relações entre variáveis.
   - Identificação das features mais correlacionadas com a penetração de internet.

5. **Preparação para Modelagem**
   - Normalização das features numéricas com `StandardScaler` e `MinMaxScaler`.
   - Redução de dimensionalidade com PCA (5 componentes).
   - Dados prontos para treino e teste do modelo.

---

## 📊 Modelagem e Avaliação

### Modelos Treinados

- **Regressão Linear**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

### Métricas de Desempenho

- **R² (R-squared):** proporção da variabilidade explicada pelo modelo.
- **RMSE (Root Mean Squared Error):** erro médio quadrático, penaliza grandes desvios.
- **MAE (Mean Absolute Error):** erro médio absoluto, interpreta facilmente a magnitude dos erros.

---

### 📁 Estrutura do Projeto

```kotlin
digital-inclusion-project/
│
├── data/
│   └── internet_adoption_clean_final.csv
│
├── notebooks/
│   ├── 00_data_ingest.ipynb
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_pca_modeling.ipynb
│   └── 04_conclusoes_storytelling.ipynb
│
├── scripts/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── modelo_final.pkl
│
├── requirements.txt
└── README.md
```

---

### 📌 Tecnologias e Bibliotecas

- Python 3.10
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

---

### 📝 Como Rodar o Projeto

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Treinar o modelo::

```bash
python scripts/train.py
```

3. Fazer previsões com novos dados:

```bash
python scripts/predict.py
```

4. Visualizar notebooks para exploração detalhada e análise.

