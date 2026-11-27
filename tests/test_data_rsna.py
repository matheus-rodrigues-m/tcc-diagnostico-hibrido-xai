"""Etapa 0.5 — integridade do RSNA Pneumonia Detection Challenge (Shih et al., 2019).

Pré-requisito: `uv run python scripts/ingest_rsna.py` (gera inventário e manifesto).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from tcc_diag.paths import CONFIGS_DIR, INTERIM_DIR, RSNA_RAW_DIR

INVENTORY = INTERIM_DIR / "rsna_inventory.csv"
pytestmark = pytest.mark.skipif(
    not INVENTORY.is_file(), reason="Rode antes: uv run python scripts/ingest_rsna.py"
)
IMG_DIR = RSNA_RAW_DIR / "stage_2_train_images"


@pytest.fixture(scope="module")
def inv() -> pd.DataFrame:
    return pd.read_csv(INVENTORY)


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    return pd.read_csv(RSNA_RAW_DIR / "stage_2_train_labels.csv")


@pytest.fixture(scope="module")
def manifest() -> dict:
    data = json.loads((CONFIGS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    return data["rsna_pneumonia"]


def test_label_files_match_manifest(manifest):
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((RSNA_RAW_DIR / name).read_bytes()).hexdigest() == digest


def test_dicom_count_and_disk_match(inv):
    assert len(inv) == 26684
    assert {p.stem for p in IMG_DIR.glob("*.dcm")} == set(inv["patientId"])


def test_images_bytes_match_manifest(inv, manifest):
    """Recalcula o SHA-256 dos 26.684 DICOMs (~3,4 GB) e compara com o manifesto."""
    lines = []
    for pid in sorted(inv["patientId"]):
        digest = hashlib.sha256((IMG_DIR / f"{pid}.dcm").read_bytes()).hexdigest()
        lines.append(f"{pid}:{digest}")
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == manifest["images_sha256"]


def test_all_dicoms_readable_and_uniform(inv):
    assert inv["readable"].all()
    assert set(zip(inv["rows"], inv["cols"], strict=True)) == {(1024, 1024)}
    assert set(inv["photometric"]) == {"MONOCHROME2"}
    assert set(inv["view"]) == {"AP", "PA"}


def test_class_counts(inv):
    """Contagens da competição (stage 2) por imagem."""
    assert inv["class"].value_counts().to_dict() == {
        "No Lung Opacity / Not Normal": 11821,
        "Normal": 8851,
        "Lung Opacity": 6012,
    }


def test_target_class_box_coherence(inv):
    """Target=1 ⇔ 'Lung Opacity' ⇔ ao menos uma caixa."""
    pos = inv["target"] == 1
    assert (inv.loc[pos, "class"] == "Lung Opacity").all()
    assert (inv.loc[~pos, "class"] != "Lung Opacity").all()
    assert (inv.loc[pos, "n_boxes"] >= 1).all()
    assert (inv.loc[~pos, "n_boxes"] == 0).all()


def test_boxes_inside_image(labels):
    b = labels.dropna(subset=["x"])
    assert len(b) == 9555
    assert ((b["x"] >= 0) & (b["y"] >= 0)).all()
    assert ((b["x"] + b["width"] <= 1024) & (b["y"] + b["height"] <= 1024)).all()


def test_no_exact_file_duplicates(inv):
    assert not inv["sha256"].duplicated().any()


def test_identifiers_do_not_link_patients(inv):
    """Registro de limitação: não há como reagrupar imagens do mesmo paciente real
    (Shih et al. 2019, p4: ~12 mil pacientes para ~30 mil imagens)."""
    for col in ("dicom_patient_id", "study_uid", "series_uid"):
        assert inv[col].nunique() == len(inv)


def test_known_age_errors_are_few(inv):
    """Idades > 100 (148–155) são erros de registro: devem ser tratadas como ausentes."""
    age = pd.to_numeric(inv["age"], errors="coerce")
    assert (age > 100).sum() == 5
