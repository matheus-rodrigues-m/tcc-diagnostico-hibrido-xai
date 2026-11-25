"""Leitura do UCI Heart Disease (Cleveland, versão processada — 14 atributos).

Fonte: Detrano et al. (1989); documentação em `heart-disease.names` (UCI ML Repository).
Esta função apenas LÊ o arquivo bruto: '?' vira NaN e os tipos são numéricos.
Nenhuma limpeza, imputação ou binarização é feita aqui — isso pertence aos
pipelines (dentro da validação cruzada) para evitar vazamento de dados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcc_diag.paths import HEART_RAW_DIR

CLEVELAND_FILE = "processed.cleveland.data"

# Ordem oficial dos 14 atributos (heart-disease.names, seção 7).
COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "num"

# Descrição clínica (PT-BR) — usada em gráficos, tabelas e na interface.
DESCRIPTIONS: dict[str, str] = {
    "age": "Idade (anos)",
    "sex": "Sexo (1 = masculino; 0 = feminino)",
    "cp": "Tipo de dor torácica (1 típica, 2 atípica, 3 não anginosa, 4 assintomática)",
    "trestbps": "Pressão arterial em repouso (mm Hg)",
    "chol": "Colesterol sérico (mg/dl)",
    "fbs": "Glicemia de jejum > 120 mg/dl (1 = sim)",
    "restecg": "ECG em repouso (0 normal, 1 anormalidade ST-T, 2 hipertrofia VE)",
    "thalach": "Frequência cardíaca máxima atingida (bpm)",
    "exang": "Angina induzida por exercício (1 = sim)",
    "oldpeak": "Depressão do segmento ST induzida por exercício (mm)",
    "slope": "Inclinação do segmento ST no pico (1 ascendente, 2 plana, 3 descendente)",
    "ca": "Nº de vasos principais coloridos por fluoroscopia (0–3)",
    "thal": "Cintilografia com tálio (3 normal, 6 defeito fixo, 7 defeito reversível)",
    "num": "Diagnóstico angiográfico (0 = < 50% estreitamento; 1–4 = > 50%)",
}

# Domínios válidos das variáveis categóricas/ordinais codificadas (heart-disease.names).
CATEGORICAL_DOMAINS: dict[str, set[int]] = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    "num": {0, 1, 2, 3, 4},
}
NUMERIC_COLUMNS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_cleveland_raw(path: Path | None = None) -> pd.DataFrame:
    """Lê o Cleveland processado com nomes de coluna oficiais; '?' -> NaN."""
    path = path or HEART_RAW_DIR / CLEVELAND_FILE
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
