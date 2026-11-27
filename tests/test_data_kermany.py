"""Etapa 0.4 — integridade do Chest X-Ray (Kermany et al., 2018; versão Kaggle).

Pré-requisito: `uv run python scripts/ingest_kermany.py` (gera inventário e manifesto).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from tcc_diag.paths import CONFIGS_DIR, INTERIM_DIR, KERMANY_RAW_DIR

INVENTORY = INTERIM_DIR / "kermany_inventory.csv"
pytestmark = pytest.mark.skipif(
    not INVENTORY.is_file(), reason="Rode antes: uv run python scripts/ingest_kermany.py"
)

EXPECTED = {
    ("train", "NORMAL"): 1341,
    ("train", "PNEUMONIA"): 3875,
    ("val", "NORMAL"): 8,
    ("val", "PNEUMONIA"): 8,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}


@pytest.fixture(scope="module")
def inv() -> pd.DataFrame:
    return pd.read_csv(INVENTORY)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((CONFIGS_DIR / "data_manifest.json").read_text(encoding="utf-8"))[
        "kermany_chest_xray"
    ]


def test_counts_match_kermany(inv):
    counts = inv.groupby(["split", "label"]).size().to_dict()
    assert counts == EXPECTED
    assert len(inv) == 5856


def test_disk_matches_inventory(inv):
    on_disk = {
        p.relative_to(KERMANY_RAW_DIR).as_posix() for p in KERMANY_RAW_DIR.glob("*/*/*.jp*g")
    }
    assert on_disk == set(inv["relpath"])


def test_no_junk_or_nested_copy():
    """__MACOSX e a cópia aninhada chest_xray/chest_xray não devem ser extraídos."""
    assert not list(KERMANY_RAW_DIR.rglob("__MACOSX"))
    assert not (KERMANY_RAW_DIR / "chest_xray").exists()


def test_images_bytes_match_manifest(inv, manifest):
    """Recalcula o SHA-256 de todas as imagens (~1,2 GB) e compara com o manifesto."""
    digests = []
    for rel in sorted(inv["relpath"]):
        digests.append(f"{rel}:{hashlib.sha256((KERMANY_RAW_DIR / rel).read_bytes()).hexdigest()}")
    assert hashlib.sha256("\n".join(digests).encode()).hexdigest() == manifest["images_sha256"]


def test_all_images_readable(inv):
    assert inv["readable"].all(), inv.loc[~inv["readable"], ["relpath", "error"]]
    assert set(inv["mode"]) <= {"L", "RGB"}


def test_every_image_has_patient_id(inv):
    assert inv["patient_id"].notna().all()


def test_no_exact_duplicates_across_splits(inv):
    """Duplicatas exatas existem (30 grupos), mas todas dentro do mesmo split."""
    splits_per_hash = inv.groupby("sha256")["split"].nunique()
    assert (splits_per_hash == 1).all()


def test_no_identical_phash_across_splits():
    """pHash idêntico (dist 0) entre splits = cópia. Dist > 0 foi inspecionado visualmente:
    pares com dist <= 4 são crianças diferentes (radiografias pediátricas são parecidas)."""
    pairs = pd.read_csv(INTERIM_DIR / "kermany_near_duplicates.csv")
    assert not ((pairs["dist"] == 0) & pairs["cross"]).any()


def test_normal_patient_ids_disjoint_train_test(inv):
    """Normais: faixas de ID de treino e teste não se sobrepõem (blocos separados)."""
    normal = inv[inv["label"] == "NORMAL"]
    train_ids = set(normal.loc[normal["split"] == "train", "patient_id"])
    test_ids = set(normal.loc[normal["split"] == "test", "patient_id"])
    assert not (train_ids & test_ids)


def test_rgb_only_in_train_pneumonia(inv):
    """Registro de um potencial atalho (shortcut): todas as RGB são pneumonia de treino.
    O pré-processamento DEVE converter tudo para tons de cinza (verificado na Fase 4)."""
    rgb = inv[inv["mode"] == "RGB"]
    assert set(zip(rgb["split"], rgb["label"], strict=True)) == {("train", "PNEUMONIA")}
