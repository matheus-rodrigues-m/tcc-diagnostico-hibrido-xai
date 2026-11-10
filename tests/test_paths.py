"""Etapa 0.2 — testes de infraestrutura: caminhos e reprodutibilidade."""

from __future__ import annotations

import random
import subprocess

import numpy as np
import pytest

from tcc_diag import paths
from tcc_diag.seed import make_rng, set_seed


# ---------------- Caminhos ----------------
def test_project_root_detected():
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (paths.PROJECT_ROOT / "PLANO_DESENVOLVIMENTO.md").is_file()


def test_required_dirs_exist():
    paths.ensure_dirs()
    missing = [d for d in paths.REQUIRED_DIRS if not d.is_dir()]
    assert not missing, f"Pastas ausentes: {missing}"


def test_paths_are_inside_project():
    for d in paths.REQUIRED_DIRS:
        assert paths.PROJECT_ROOT in d.parents, f"{d} fora da raiz do projeto"


def test_literature_folder_present():
    assert paths.LITERATURE_DIR.is_dir()
    assert (paths.LITERATURE_DIR / "PROPOSTA-TRABALHO-TCCI.pdf").is_file()


def test_data_dir_env_override(monkeypatch, tmp_path):
    """TCC_DATA_DIR deve redirecionar a pasta de dados (cenário Kaggle/Colab)."""
    import importlib

    monkeypatch.setenv("TCC_DATA_DIR", str(tmp_path))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA_DIR == tmp_path.resolve()
        assert reloaded.RAW_DIR == tmp_path.resolve() / "raw"
    finally:
        monkeypatch.delenv("TCC_DATA_DIR")
        importlib.reload(paths)


# ---------------- Sementes ----------------
def _draw():
    return random.random(), np.random.rand(3).tolist()


def test_seed_reproducible_python_numpy():
    set_seed(123)
    a = _draw()
    set_seed(123)
    b = _draw()
    set_seed(124)
    c = _draw()
    assert a == b, "mesma semente deve gerar a mesma sequência"
    assert a != c, "sementes diferentes devem gerar sequências diferentes"


def test_make_rng_independent():
    assert np.array_equal(make_rng(7).integers(0, 1000, 10), make_rng(7).integers(0, 1000, 10))


def test_seed_reproducible_torch():
    torch = pytest.importorskip("torch")
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)
    if torch.cuda.is_available():
        set_seed(42)
        g1 = torch.randn(5, device="cuda")
        set_seed(42)
        g2 = torch.randn(5, device="cuda")
        assert torch.equal(g1, g2)


# ---------------- Git ----------------
@pytest.mark.parametrize(
    "path, should_ignore",
    [
        ("kaggle_token.txt", True),
        ("explore/PROPOSTA-TRABALHO-TCCI.pdf", True),
        ("data/raw/heart_disease/processed.cleveland.data", True),
        ("models/xray/resnet50.pt", True),
        (".venv/pyvenv.cfg", True),
        ("data/interim/splits/kermany_splits.csv", False),
        ("src/tcc_diag/paths.py", False),
        ("PLANO_DESENVOLVIMENTO.md", False),
    ],
)
def test_gitignore_rules(path, should_ignore):
    """Credenciais, dados, pesos e literatura nunca devem entrar no repositório."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=paths.PROJECT_ROOT, capture_output=True
    )
    assert (result.returncode == 0) == should_ignore
