"""Caminhos centralizados do projeto.

Toda leitura/escrita de arquivos nos notebooks e módulos deve usar as constantes daqui,
nunca caminhos fixos. Isso garante que o mesmo código rode no PC local e em ambientes
de nuvem (Kaggle/Colab).

Sobrescritas por variável de ambiente:
    TCC_PROJECT_ROOT  raiz do projeto (padrão: detectada a partir deste arquivo)
    TCC_DATA_DIR      pasta de dados (padrão: <raiz>/data) — útil no Kaggle, onde os
                      datasets ficam em /kaggle/input
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Sobe a árvore de diretórios a partir deste arquivo até achar o pyproject.toml."""
    env = os.environ.get("TCC_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Raiz do projeto (pyproject.toml) não encontrada.")


PROJECT_ROOT: Path = _find_project_root()

# --- Dados ---
DATA_DIR: Path = Path(os.environ.get("TCC_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = INTERIM_DIR / "splits"

HEART_RAW_DIR: Path = RAW_DIR / "heart_disease"
KERMANY_RAW_DIR: Path = RAW_DIR / "chest_xray_kermany"
RSNA_RAW_DIR: Path = RAW_DIR / "rsna_pneumonia"

# --- Literatura local (somente leitura) ---
LITERATURE_DIR: Path = PROJECT_ROOT / "explore"

# --- Artefatos ---
MODELS_DIR: Path = PROJECT_ROOT / "models"
HEART_MODELS_DIR: Path = MODELS_DIR / "heart"
XRAY_MODELS_DIR: Path = MODELS_DIR / "xray"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Pastas que o projeto cria/espera existir (as de dados brutos são criadas pelos downloads).
REQUIRED_DIRS: tuple[Path, ...] = (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR,
    MODELS_DIR, HEART_MODELS_DIR, XRAY_MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, METRICS_DIR, CONFIGS_DIR, NOTEBOOKS_DIR,
)


def ensure_dirs() -> None:
    """Cria (se faltarem) todas as pastas de trabalho do projeto."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
