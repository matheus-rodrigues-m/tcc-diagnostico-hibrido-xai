"""Etapa 0.5 — Ingestão e integridade do RSNA Pneumonia Detection Challenge.

Referência: Shih et al. (2019), Radiology: AI — anotações de 18 radiologistas sobre um
subconjunto do NIH ChestX-ray14 (adultos e alguns menores, incidências AP e PA).

Passos:
 1. Baixa o .zip da competição (requer aceitar as regras no Kaggle) e registra o SHA-256.
 2. Extrai SOMENTE as imagens de treino (rotuladas) e os CSVs de rótulos; as 3.000 imagens
    de teste da competição não têm rótulo e são ignoradas.
 3. Valida os CSVs: coerência Target × classe detalhada, caixas dentro da imagem.
 4. Lê todos os DICOMs (pydicom): legibilidade, dimensões, metadados (idade, sexo, AP/PA),
    UIDs que poderiam ligar imagens do mesmo paciente, SHA-256 e pHash.
 5. Atualiza configs/data_manifest.json e imprime o relatório.

Uso:  uv run python scripts/ingest_rsna.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import imagehash
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from PIL import Image

from tcc_diag.paths import CONFIGS_DIR, INTERIM_DIR, RSNA_RAW_DIR

COMPETITION = "rsna-pneumonia-detection-challenge"
ZIP_PATH = RSNA_RAW_DIR / f"{COMPETITION}.zip"
IMG_DIR = RSNA_RAW_DIR / "stage_2_train_images"
LABELS_CSV = RSNA_RAW_DIR / "stage_2_train_labels.csv"
CLASS_CSV = RSNA_RAW_DIR / "stage_2_detailed_class_info.csv"
INVENTORY = INTERIM_DIR / "rsna_inventory.csv"
MANIFEST = CONFIGS_DIR / "data_manifest.json"

N_TRAIN_IMAGES = 26684  # documentação da competição (stage 2)
CLASSES = {"Lung Opacity", "Normal", "No Lung Opacity / Not Normal"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- 1–2
def download_and_extract() -> None:
    if not ZIP_PATH.is_file():
        kaggle = Path(sys.executable).parent / (
            "kaggle.exe" if sys.platform == "win32" else "kaggle"
        )
        RSNA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        print("[1] baixando do Kaggle…")
        cmd = [str(kaggle), "competitions", "download", "-c", COMPETITION, "-p", str(RSNA_RAW_DIR)]
        subprocess.run(cmd, check=True)
    else:
        print(f"[1] .zip já presente ({ZIP_PATH.stat().st_size / 1e9:.2f} GB) — download pulado")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        train = [n for n in names if n.startswith("stage_2_train_images/") and n.endswith(".dcm")]
        test = [n for n in names if n.startswith("stage_2_test_images/") and n.endswith(".dcm")]
        csvs = [n for n in names if n in (LABELS_CSV.name, CLASS_CSV.name)]
        print(
            f"[2] no .zip: {len(train)} DICOMs de treino, {len(test)} de teste (ignorados), "
            f"CSVs: {csvs}"
        )
        present = len(list(IMG_DIR.glob("*.dcm"))) if IMG_DIR.is_dir() else 0
        if present == len(train) and LABELS_CSV.is_file() and CLASS_CSV.is_file():
            print(f"      já extraído ({present} DICOMs) — extração pulada")
            return
        shutil.rmtree(IMG_DIR, ignore_errors=True)
        for n in train + csvs:
            zf.extract(n, RSNA_RAW_DIR)
        print(f"      extraídos {len(train)} DICOMs + {len(csvs)} CSVs")


# --------------------------------------------------------------------------- 3
def validate_labels() -> tuple[pd.DataFrame, dict]:
    lab = pd.read_csv(LABELS_CSV)
    cls = pd.read_csv(CLASS_CSV)
    rep: dict = {
        "label_rows": len(lab),
        "class_rows": len(cls),
        "unique_ids_labels": lab["patientId"].nunique(),
        "unique_ids_class": cls["patientId"].nunique(),
    }
    # Uma classe por patientId (as linhas repetidas correspondem às várias caixas)
    cls_u = cls.drop_duplicates()
    rep["ids_with_conflicting_class"] = int(cls_u["patientId"].duplicated().sum())
    per_id = (
        lab.groupby("patientId")
        .agg(
            target=("Target", "max"),
            target_min=("Target", "min"),
            n_boxes=("x", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .merge(cls_u, on="patientId", how="outer", indicator=True)
    )
    rep["ids_only_in_one_csv"] = int((per_id["_merge"] != "both").sum())
    rep["ids_mixed_target"] = int((per_id["target"] != per_id["target_min"]).sum())
    rep["class_counts"] = per_id["class"].value_counts().to_dict()
    rep["target_counts"] = per_id["target"].value_counts().to_dict()
    pos, neg = per_id["target"] == 1, per_id["target"] == 0
    rep["pos_not_lung_opacity"] = int((pos & (per_id["class"] != "Lung Opacity")).sum())
    rep["neg_is_lung_opacity"] = int((neg & (per_id["class"] == "Lung Opacity")).sum())
    rep["pos_without_box"] = int((pos & (per_id["n_boxes"] == 0)).sum())
    rep["neg_with_box"] = int((neg & (per_id["n_boxes"] > 0)).sum())
    rep["unknown_classes"] = sorted(set(per_id["class"].dropna()) - CLASSES)
    boxes = lab.dropna(subset=["x"])
    rep["n_boxes"] = len(boxes)
    rep["boxes_per_positive"] = per_id.loc[pos, "n_boxes"].value_counts().sort_index().to_dict()
    inside = (
        (boxes["x"] >= 0)
        & (boxes["y"] >= 0)
        & (boxes["width"] > 0)
        & (boxes["height"] > 0)
        & (boxes["x"] + boxes["width"] <= 1024)
        & (boxes["y"] + boxes["height"] <= 1024)
    )
    rep["boxes_outside_image"] = int((~inside).sum())
    rep["box_area_frac_median"] = float((boxes["width"] * boxes["height"] / 1024**2).median())
    return per_id.drop(columns=["_merge", "target_min"]), rep


# --------------------------------------------------------------------------- 4
def describe_dicom(path: Path) -> dict:
    row = {
        "patientId": path.stem,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "readable": True,
        "error": None,
    }
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        row.update(
            rows=int(ds.Rows),
            cols=int(ds.Columns),
            bits=int(ds.BitsStored),
            photometric=str(ds.PhotometricInterpretation),
            dicom_patient_id=str(ds.get("PatientID", "")),
            age=str(ds.get("PatientAge", "")),
            sex=str(ds.get("PatientSex", "")),
            view=str(ds.get("ViewPosition", "")),
            body_part=str(ds.get("BodyPartExamined", "")),
            modality=str(ds.get("Modality", "")),
            study_uid=str(ds.get("StudyInstanceUID", "")),
            series_uid=str(ds.get("SeriesInstanceUID", "")),
            pix_min=int(arr.min()),
            pix_max=int(arr.max()),
            phash=str(imagehash.phash(Image.fromarray(arr).convert("L"))),
        )
    except Exception as exc:  # noqa: BLE001
        row.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    return row


def build_inventory(per_id: pd.DataFrame) -> pd.DataFrame:
    files = sorted(IMG_DIR.glob("*.dcm"))
    print(f"[4] lendo {len(files)} DICOMs (pixels + metadados + SHA-256 + pHash)…")
    rows = Parallel(n_jobs=-1, batch_size=64)(delayed(describe_dicom)(p) for p in files)
    inv = pd.DataFrame(rows).merge(per_id, on="patientId", how="left")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(INVENTORY, index=False)
    return inv


# --------------------------------------------------------------------------- 5
def update_manifest(entry: dict) -> bool:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    old = manifest.get("rsna_pneumonia", {})
    if all(old.get(k) == entry[k] for k in ("zip_sha256", "images_sha256", "labels_sha256")):
        return False
    entry["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["rsna_pneumonia"] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    download_and_extract()
    zip_sha = sha256_file(ZIP_PATH)
    print("[3] validando CSVs de rótulos…")
    per_id, rep = validate_labels()
    inv = build_inventory(per_id)
    ok = inv[inv["readable"]]

    print("\n================ RELATÓRIO RSNA ================")
    print("Rótulos (CSV):")
    for k, v in rep.items():
        print(f"  {k:<28}: {v}")

    print(
        f"\nDICOMs: {len(inv)} (esperado {N_TRAIN_IMAGES}) | "
        f"ilegíveis: {int((~inv['readable']).sum())}"
    )
    print(f"  sem rótulo (DICOM sem linha no CSV): {int(inv['target'].isna().sum())}")
    print(f"  dimensões: {ok.groupby(['rows', 'cols']).size().to_dict()}")
    print(
        f"  bits armazenados: {ok['bits'].value_counts().to_dict()} | "
        f"fotométrico: {ok['photometric'].value_counts().to_dict()}"
    )
    print(f"  pixel min/max: {ok['pix_min'].min()}–{ok['pix_max'].max()}")
    print(
        f"  modalidade: {ok['modality'].value_counts().to_dict()} | "
        f"parte do corpo: {ok['body_part'].value_counts().to_dict()}"
    )
    print(f"  incidência: {ok['view'].value_counts().to_dict()}")
    print(f"  sexo: {ok['sex'].value_counts().to_dict()}")
    age = pd.to_numeric(ok["age"], errors="coerce")
    print(
        f"  idade: mediana {age.median():.0f} | min {age.min():.0f} | max {age.max():.0f} | "
        f"<18: {int((age < 18).sum())} | >100: {int((age > 100).sum())} "
        f"{sorted(age[age > 100].astype(int).tolist())}"
    )
    print("  incidência × alvo (% positivos):")
    for v, g in ok.groupby("view"):
        print(f"    {v}: n={len(g):>5}  positivos={g['target'].mean():.1%}")

    print("\nIdentificadores (dá para agrupar por paciente real?):")
    print(
        "  DICOM PatientID == nome do arquivo: "
        f"{(ok['dicom_patient_id'] == ok['patientId']).mean():.1%}"
    )
    for col in ("dicom_patient_id", "study_uid", "series_uid"):
        print(f"  {col:<17} únicos: {ok[col].nunique():>6} / {len(ok)}")

    exact = ok[ok.duplicated("sha256", keep=False)]
    print(f"\nDuplicatas exatas de arquivo (SHA-256): {exact['sha256'].nunique()} grupos")
    ph = ok[ok.duplicated("phash", keep=False)].sort_values("phash")
    print(f"pHash idêntico (dist 0): {ph['phash'].nunique()} grupos, {len(ph)} imagens")
    if len(ph):
        ph[["phash", "patientId", "target", "class", "age", "sex", "view"]].to_csv(
            INTERIM_DIR / "rsna_identical_phash.csv", index=False
        )
        tgt_conf = ph.groupby("phash")["target"].nunique().gt(1).sum()
        print(
            f"  grupos com rótulos conflitantes: {int(tgt_conf)} "
            "(detalhes: rsna_identical_phash.csv)"
        )

    images_sha = hashlib.sha256(
        "\n".join(
            f"{p}:{h}" for p, h in sorted(zip(inv["patientId"], inv["sha256"], strict=True))
        ).encode()
    ).hexdigest()
    labels_sha = hashlib.sha256(
        (sha256_file(LABELS_CSV) + sha256_file(CLASS_CSV)).encode()
    ).hexdigest()
    changed = update_manifest(
        {
            "source": f"Kaggle competition: {COMPETITION} (Shih et al., 2019)",
            "url": f"https://www.kaggle.com/c/{COMPETITION}",
            "local_dir": "data/raw/rsna_pneumonia",
            "zip_sha256": zip_sha,
            "zip_bytes": ZIP_PATH.stat().st_size,
            "n_train_dicoms": len(inv),
            "files": {
                LABELS_CSV.name: sha256_file(LABELS_CSV),
                CLASS_CSV.name: sha256_file(CLASS_CSV),
            },
            "labels_sha256": labels_sha,
            "images_sha256": images_sha,
            "class_counts": rep["class_counts"],
        }
    )
    print(
        f"\nManifesto {'atualizado' if changed else 'inalterado (hashes idênticos)'}; "
        f"inventário: {INVENTORY.name}"
    )

    critical = [
        len(inv) == N_TRAIN_IMAGES,
        inv["readable"].all(),
        inv["target"].notna().all(),
        rep["pos_not_lung_opacity"] == 0,
        rep["neg_is_lung_opacity"] == 0,
        rep["pos_without_box"] == 0,
        rep["neg_with_box"] == 0,
        rep["ids_with_conflicting_class"] == 0,
        rep["boxes_outside_image"] == 0,
    ]
    return 0 if all(critical) else 1


if __name__ == "__main__":
    sys.exit(main())
