"""Reprodutibilidade: fixação de sementes aleatórias.

Chamar `set_seed(SEED)` no início de todo notebook/experimento. O valor usado deve ser
registrado junto às métricas (Sculley et al., 2015, p. 8 — dívida de reprodutibilidade).

Observação: mesmo com sementes fixas, algumas operações de GPU são não determinísticas.
Com `deterministic=True` o cuDNN usa apenas algoritmos determinísticos (um pouco mais
lentos); diferenças residuais da ordem de 1e-6 entre execuções ainda podem ocorrer.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Fixa as sementes de Python, NumPy e PyTorch (CPU e GPU). Retorna a semente usada."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # módulos tabulares não exigem torch
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Necessário para determinismo de algumas operações cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Gerador NumPy independente — preferível ao estado global para splits e amostragens."""
    return np.random.default_rng(seed)
