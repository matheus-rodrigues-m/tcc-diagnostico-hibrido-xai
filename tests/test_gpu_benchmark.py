"""Etapa 0.6 — sanidade do benchmark de GPU e da configuração de treino escolhida.

Pré-requisito: `uv run python scripts/benchmark_gpu.py` (gera reports/metrics/gpu_benchmark.json).
"""

from __future__ import annotations

import json

import pytest

from tcc_diag.paths import METRICS_DIR

BENCH = METRICS_DIR / "gpu_benchmark.json"
pytestmark = pytest.mark.skipif(
    not BENCH.is_file(), reason="Rode antes: uv run python scripts/benchmark_gpu.py"
)
ARCHS = ("resnet50", "densenet121", "efficientnet_b0")
WORK_BATCH = 32  # batch comum às três redes (comparação justa)


@pytest.fixture(scope="module")
def bench() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


def test_all_architectures_measured(bench):
    assert set(bench["results"]) == set(ARCHS)


@pytest.mark.parametrize("arch", ARCHS)
def test_work_batch_fits_with_margin(bench, arch):
    """Batch 32 deve caber com folga (< 60% da VRAM) para acomodar Grad-CAM e imagens maiores."""
    r = bench["results"][arch]
    assert r["max_train_batch"] >= WORK_BATCH
    assert r["work_batch"] == WORK_BATCH
    assert r["train_peak_vram_GB"] < 0.6 * bench["vram_GB"]


@pytest.mark.parametrize("arch", ARCHS)
def test_training_throughput_sane(bench, arch):
    """Guarda contra a regressão channels_last+BatchNorm (23 img/s na ResNet-50).
    Treino deve ser no máximo ~5x mais lento que a inferência."""
    r = bench["results"][arch]
    assert r["train_img_s"] > 100
    assert r["infer_img_s"] / r["train_img_s"] < 5


def test_settings_recorded(bench):
    s = bench["settings"]
    assert s["amp"] == "fp16" and s["deterministic"] is True
    assert s["channels_last"] is False
    assert s["vram_fraction_cap"] <= 0.95


def test_protocol_fits_locally(bench):
    """Decisão da etapa: o protocolo cabe na GPU local (limite superior < 24 h)."""
    assert bench["protocol_estimate"]["total_hours_upper_bound"] < 24
