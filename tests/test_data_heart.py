"""Etapa 0.3 — integridade do UCI Heart Disease (Cleveland).

Valores esperados conferidos contra a documentação oficial (`heart-disease.names`)
e Detrano et al. (1989). Se os dados ainda não foram ingeridos, os testes são pulados
com instrução de como gerá-los.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tcc_diag.paths import CONFIGS_DIR, HEART_RAW_DIR
from tcc_diag.tabular.data import (
    CATEGORICAL_DOMAINS,
    CLEVELAND_FILE,
    COLUMNS,
    NUMERIC_COLUMNS,
    load_cleveland_raw,
)

pytestmark = pytest.mark.skipif(
    not (HEART_RAW_DIR / CLEVELAND_FILE).is_file(),
    reason="Rode antes: uv run python scripts/ingest_heart.py",
)


@pytest.fixture(scope="module")
def df():
    return load_cleveland_raw()


def test_checksum_matches_manifest():
    """O arquivo em disco deve ser byte a byte o registrado no manifesto versionado."""
    manifest = json.loads((CONFIGS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["uci_heart_disease"]["files"].items():
        digest = hashlib.sha256((HEART_RAW_DIR / name).read_bytes()).hexdigest()
        assert digest == info["sha256"], f"{name} foi alterado desde a ingestão"


def test_shape_and_columns(df):
    assert df.shape == (303, 14)
    assert list(df.columns) == COLUMNS


def test_missing_only_in_ca_and_thal(df):
    missing = df.isna().sum()
    assert missing[missing > 0].to_dict() == {"ca": 4, "thal": 2}
    assert df.isna().any(axis=1).sum() == 6


def test_target_distribution(df):
    assert df["num"].value_counts().sort_index().to_dict() == {0: 164, 1: 55, 2: 36, 3: 35, 4: 13}
    assert (df["num"] > 0).sum() == 139  # presença (1–4) vs ausência (0)


@pytest.mark.parametrize("col", list(CATEGORICAL_DOMAINS))
def test_categorical_domains(df, col):
    values = df[col].dropna()
    assert np.all(values == values.round()), f"{col} deveria ser inteiro"
    assert set(values.astype(int)) <= CATEGORICAL_DOMAINS[col]


def test_numeric_ranges_physiologically_plausible(df):
    """Limites amplos de plausibilidade — detectam erros de digitação/unidade, não outliers."""
    limits = {
        "age": (18, 100), "trestbps": (70, 250), "chol": (100, 700),
        "thalach": (60, 220), "oldpeak": (0, 7),
    }
    for col in NUMERIC_COLUMNS:
        lo, hi = limits[col]
        assert df[col].between(lo, hi).all(), f"{col} fora de [{lo}, {hi}]"


def test_no_exact_duplicates(df):
    """Contraste com o heart.csv do Kaggle (1025 linhas, ~723 duplicatas)."""
    assert not df.duplicated().any()
