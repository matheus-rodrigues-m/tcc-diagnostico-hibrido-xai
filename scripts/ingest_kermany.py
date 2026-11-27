"""Etapa 0.4 — Ingestão e integridade do Chest X-Ray (Kermany et al., 2018; versão Kaggle).

Passos:
 1. Baixa o .zip do Kaggle (se ainda não existir) e registra seu SHA-256.
 2. O .zip contém lixo do macOS (__MACOSX) e uma cópia aninhada (chest_xray/chest_xray/).
    O script identifica as cópias, verifica (por CRC32) se são idênticas e extrai UMA só
    para data/raw/chest_xray_kermany/{train,val,test}/{NORMAL,PNEUMONIA}/.
 3. Gera um inventário por imagem (data/interim/kermany_inventory.csv): split, classe,
    ID do paciente (extraído do nome do arquivo), subtipo, dimensões, modo de cor,
    SHA-256 e hash perceptual (pHash, 64 bits).
 4. Procura duplicatas exatas (SHA-256) e quase-duplicatas (distância de Hamming do pHash)
    dentro e entre splits — risco de vazamento treino↔teste.
 5. Atualiza configs/data_manifest.json e imprime o relatório.

Uso:  uv run python scripts/ingest_kermany.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import imagehash
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from tcc_diag.paths import CONFIGS_DIR, INTERIM_DIR, KERMANY_RAW_DIR

KAGGLE_REF = "paultimothymooney/chest-xray-pneumonia"
ZIP_PATH = KERMANY_RAW_DIR / "chest-xray-pneumonia.zip"
INVENTORY = INTERIM_DIR / "kermany_inventory.csv"
MANIFEST = CONFIGS_DIR / "data_manifest.json"
SPLITS, LABELS = ("train", "val", "test"), ("NORMAL", "PNEUMONIA")

# Contagens esperadas na versão Kaggle (Kermany 2018, p7: treino 5232 = 1349 N + 3883 P;
# o Kaggle moveu 8+8 imagens do treino para "val"; teste 624 = 234 N + 390 P).
EXPECTED = {
    ("train", "NORMAL"): 1341,
    ("train", "PNEUMONIA"): 3875,
    ("val", "NORMAL"): 8,
    ("val", "PNEUMONIA"): 8,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}

# Padrões de nome: pneumonia "person{ID}_{bacteria|virus}_{n}.jpeg";
# normal "IM-{ID}-{seq}.jpeg", "NORMAL2-IM-{ID}-{seq}.jpeg" e, só no teste,
# "IM-{ID}-{seq}-{sub}.jpeg" (várias imagens do mesmo exame).
RE_PNEU = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)(?:_\d+)?\.jpe?g$", re.I)
RE_NORM = re.compile(r"^(NORMAL2-)?IM-(\d+)-(\d+)(?:-(\d+))?\.jpe?g$", re.I)
IMG_ENTRY = re.compile(
    r"^(?P<root>(?:[^/]+/)*?)(?P<split>train|val|test)/(?P<label>NORMAL|PNEUMONIA)/(?P<name>[^/]+\.jpe?g)$",
    re.I,
)


# --------------------------------------------------------------------------- utilidades
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_patient(name: str, label: str) -> tuple[str | None, str | None]:
    """Retorna (patient_id, subtipo). IDs ganham prefixo para não colidirem entre classes."""
    if label == "PNEUMONIA":
        m = RE_PNEU.match(name)
        return (f"P{int(m.group(1))}", m.group(2).lower()) if m else (None, None)
    m = RE_NORM.match(name)
    if not m:
        return None, None
    prefix = "N2-" if m.group(1) else "N-"
    return f"{prefix}{int(m.group(2))}", "normal"


# --------------------------------------------------------------------------- 1. download
def download() -> None:
    if ZIP_PATH.is_file():
        print(f"[1] .zip já presente ({ZIP_PATH.stat().st_size / 1e9:.2f} GB) — download pulado")
        return
    kaggle = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    KERMANY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[1] baixando do Kaggle…")
    subprocess.run(
        [str(kaggle), "datasets", "download", "-d", KAGGLE_REF, "-p", str(KERMANY_RAW_DIR)],
        check=True,
    )


# --------------------------------------------------------------------------- 2. extração
def extract() -> dict:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        copies: dict[str, dict[str, int]] = defaultdict(dict)  # raiz -> {split/label/nome: crc}
        for info in zf.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            m = IMG_ENTRY.match(info.filename)
            if m:
                key = f"{m['split']}/{m['label']}/{m['name']}"
                copies[m["root"]][key] = info.CRC
        roots = sorted(copies, key=len)
        print(f"[2] cópias do dataset dentro do .zip: {[r or '<raiz>' for r in roots]}")
        for r in roots:
            print(f"      {r or '<raiz>':<25} {len(copies[r])} imagens")
        canonical = roots[0]
        identical = {r: copies[r] == copies[canonical] for r in roots[1:]}
        for r, same in identical.items():
            print(f"      '{r}' idêntica à canônica (nomes + CRC32)? {same}")

        target_ok = all((KERMANY_RAW_DIR / s / lab).is_dir() for s in SPLITS for lab in LABELS)
        n_present = sum(1 for _ in KERMANY_RAW_DIR.glob("*/*/*.jp*g")) if target_ok else 0
        if n_present == len(copies[canonical]):
            print(f"      já extraído ({n_present} imagens) — extração pulada")
        else:
            for s in SPLITS:
                shutil.rmtree(KERMANY_RAW_DIR / s, ignore_errors=True)
            for key in copies[canonical]:
                src = canonical + key
                dst = KERMANY_RAW_DIR / PurePosixPath(key)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(src) as fsrc, dst.open("wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
            print(f"      extraídas {len(copies[canonical])} imagens de '{canonical or '<raiz>'}'")
    return {
        "copies_in_zip": {r or "<root>": len(c) for r, c in copies.items()},
        "copies_identical": identical,
        "canonical_root": canonical or "<root>",
    }


# --------------------------------------------------------------------------- 3. inventário
def describe_image(path: Path) -> dict:
    split, label, name = path.parts[-3], path.parts[-2], path.name
    pid, subtype = parse_patient(name, label)
    row = {
        "relpath": f"{split}/{label}/{name}",
        "split": split,
        "label": label,
        "patient_id": pid,
        "subtype": subtype,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "readable": True,
        "error": None,
    }
    try:
        with Image.open(path) as im:
            im.verify()  # checa estrutura do arquivo
        with Image.open(path) as im:
            im.load()  # decodifica de fato os pixels
            row.update(
                width=im.width,
                height=im.height,
                mode=im.mode,
                phash=str(imagehash.phash(im.convert("L"))),
            )
    except Exception as exc:  # noqa: BLE001
        row.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    return row


def build_inventory() -> pd.DataFrame:
    files = sorted(KERMANY_RAW_DIR.glob("*/*/*.jp*g"))
    print(f"[3] inventariando {len(files)} imagens (SHA-256 + pHash)…")
    rows = Parallel(n_jobs=-1, batch_size=32)(delayed(describe_image)(p) for p in files)
    df = pd.DataFrame(rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY, index=False)
    return df


# --------------------------------------------------------------------------- 4. duplicatas
def hamming_pairs(df: pd.DataFrame, max_dist: int) -> pd.DataFrame:
    """Todos os pares (i<j) com distância de Hamming do pHash <= max_dist (vetorizado)."""
    h = np.array([int(x, 16) for x in df["phash"]], dtype=np.uint64)
    out = []
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i + 1 :], h[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            out.append((i, i + 1 + j, int(d[j])))
    pairs = pd.DataFrame(out, columns=["i", "j", "dist"])
    for side in ("i", "j"):
        sub = df.iloc[pairs[side]].reset_index(drop=True)
        for col in ("relpath", "split", "label", "patient_id"):
            pairs[f"{col}_{side}"] = sub[col]
    return pairs


# --------------------------------------------------------------------------- 5. manifesto
def update_manifest(entry: dict) -> bool:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    old = manifest.get("kermany_chest_xray", {})
    if (
        old.get("zip_sha256") == entry["zip_sha256"]
        and old.get("images_sha256") == entry["images_sha256"]
    ):
        return False
    entry["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["kermany_chest_xray"] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    download()
    zip_sha = sha256_file(ZIP_PATH)
    zip_info = extract()
    df = build_inventory()
    ok = df[df["readable"]]

    print("\n================ RELATÓRIO KERMANY ================")
    counts = df.groupby(["split", "label"]).size()
    print("Contagens por split/classe (obtido | esperado):")
    count_ok = True
    for (s, lab), exp in EXPECTED.items():
        got = int(counts.get((s, lab), 0))
        count_ok &= got == exp
        print(f"  {s:<5} {lab:<9} {got:>5} | {exp:>5}  {'OK' if got == exp else 'DIVERGE'}")
    print(f"  total {len(df)}")

    print(f"\nImagens ilegíveis: {int((~df['readable']).sum())}")
    print(f"Modos de cor: {ok['mode'].value_counts().to_dict()}")
    for col, name in (("width", "Largura"), ("height", "Altura ")):
        s = ok[col]
        print(f"{name}  px: min {s.min()}  mediana {int(s.median())}  max {s.max()}")
    ar = ok["width"] / ok["height"]
    print(f"Razão L/A  : min {ar.min():.2f}  mediana {ar.median():.2f}  max {ar.max():.2f}")

    unparsed = df[df["patient_id"].isna()]
    n_ok = len(df) - len(unparsed)
    print(f"\nID de paciente extraído: {n_ok}/{len(df)}  (sem padrão: {len(unparsed)})")
    for r in unparsed["relpath"].head(10):
        print(f"    sem padrão: {r}")
    subtypes = df.loc[df["label"] == "PNEUMONIA", "subtype"].value_counts().to_dict()
    print(f"Subtipos de pneumonia: {subtypes}")
    pts = df.groupby(["split", "label"])["patient_id"].nunique()
    print("Pacientes únicos (imagens/paciente):")
    for (s, lab), n in pts.items():
        print(f"  {s:<5} {lab:<9} {n:>5} pacientes  ({counts[(s, lab)] / n:.2f} img/pac)")
    ids = {s: set(df.loc[df["split"] == s, "patient_id"].dropna()) for s in SPLITS}
    id_overlap = {
        f"{a}-{b}": len(ids[a] & ids[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    print(f"IDs de paciente coincidentes entre splits: {id_overlap}")

    exact = (
        df[df.duplicated("sha256", keep=False)]
        .groupby("sha256")["split"]
        .apply(lambda s: "+".join(sorted(s)))
    )
    print(f"\nDuplicatas EXATAS (SHA-256): {exact.size} grupos  {exact.value_counts().to_dict()}")
    pairs = hamming_pairs(ok.reset_index(drop=True), max_dist=6)
    pairs["cross"] = pairs["split_i"] != pairs["split_j"]
    pairs["same_patient"] = pairs["patient_id_i"] == pairs["patient_id_j"]
    split_pair = pairs[["split_i", "split_j"]].apply(lambda r: frozenset(r), axis=1)
    pairs["train_test"] = split_pair == frozenset({"train", "test"})
    pairs.to_csv(INTERIM_DIR / "kermany_near_duplicates.csv", index=False)
    print("Quase-duplicatas por pHash (pares, distância de Hamming):")
    for d in (0, 2, 4, 6):
        sub = pairs[pairs["dist"] <= d]
        print(
            f"  dist<={d}: {len(sub):>4} pares | entre splits: {int(sub['cross'].sum()):>3}"
            f" | mesmo paciente: {int(sub['same_patient'].sum()):>3}"
            f" | treino↔teste: {int(sub['train_test'].sum()):>3}"
        )

    images_sha = hashlib.sha256(
        "\n".join(
            f"{r}:{h}" for r, h in sorted(zip(df["relpath"], df["sha256"], strict=True))
        ).encode()
    ).hexdigest()
    changed = update_manifest(
        {
            "source": f"Kaggle: {KAGGLE_REF} (Kermany et al., 2018, Cell)",
            "url": f"https://www.kaggle.com/datasets/{KAGGLE_REF}",
            "local_dir": "data/raw/chest_xray_kermany",
            "zip_sha256": zip_sha,
            "zip_bytes": ZIP_PATH.stat().st_size,
            **zip_info,
            "n_images": len(df),
            "counts": {f"{s}/{lab}": int(counts.get((s, lab), 0)) for s, lab in EXPECTED},
            "images_sha256": images_sha,
        }
    )
    print(
        f"\nManifesto {'atualizado' if changed else 'inalterado (hashes idênticos)'}; "
        f"inventário: {INVENTORY.name}; pares pHash: kermany_near_duplicates.csv"
    )
    return 0 if count_ok and df["readable"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
