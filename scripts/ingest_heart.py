"""Etapa 0.3 — Ingestão do UCI Heart Disease.

Copia os arquivos originais de `explore/5. Datasets Benchmark/heart+disease/` para
`data/raw/heart_disease/` (sem alterá-los), calcula o SHA-256 de cada um e registra
tudo em `configs/data_manifest.json` (versionado no git). Ao final, imprime um
relatório de integridade do Cleveland.

Uso:  uv run python scripts/ingest_heart.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from tcc_diag.paths import CONFIGS_DIR, HEART_RAW_DIR, LITERATURE_DIR
from tcc_diag.tabular.data import CATEGORICAL_DOMAINS, COLUMNS, load_cleveland_raw

SOURCE_DIR = LITERATURE_DIR / "5. Datasets Benchmark" / "heart+disease"
FILES = [
    "processed.cleveland.data",   # dataset principal (303 × 14)
    "processed.hungarian.data",   # validação externa opcional (etapa 2.5)
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",        # documentação oficial dos atributos
    "WARNING",                    # aviso da UCI sobre o cleveland.data corrompido
]
MANIFEST = CONFIGS_DIR / "data_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(key: str, entry: dict) -> bool:
    """Mescla a entrada no manifesto (outros datasets serão adicionados nas etapas 0.4/0.5).

    Idempotente: se os arquivos (hashes) não mudaram, o manifesto não é reescrito — assim
    reexecutar o script não gera alterações espúrias no git. Retorna True se houve mudança.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if manifest.get(key, {}).get("files") == entry["files"]:
        return False
    entry["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest[key] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    HEART_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files_info = {}
    for name in FILES:
        src, dst = SOURCE_DIR / name, HEART_RAW_DIR / name
        if not src.is_file():
            print(f"ERRO: arquivo de origem ausente: {src}")
            return 1
        shutil.copy2(src, dst)
        digest = sha256(dst)
        assert digest == sha256(src), f"cópia corrompida: {name}"
        files_info[name] = {"sha256": digest, "bytes": dst.stat().st_size}
        print(f"  copiado  {name:<28} {dst.stat().st_size:>7} B  sha256={digest[:16]}…")

    changed = update_manifest("uci_heart_disease", {
        "source": "UCI ML Repository — Heart Disease (id 45); cópia local em explore/",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "local_dir": "data/raw/heart_disease",
        "files": files_info,
    })
    rel = MANIFEST.relative_to(MANIFEST.parents[1])
    print(f"\nManifesto {'atualizado' if changed else 'inalterado (hashes idênticos)'}: {rel}")

    # ---------------- Relatório de integridade do Cleveland ----------------
    df = load_cleveland_raw()
    print("\n--- Cleveland (processed) ---")
    print(f"  shape                 : {df.shape}")
    print(f"  colunas               : {list(df.columns) == COLUMNS}")
    print(f"  valores ausentes      : {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    print(f"  linhas incompletas    : {int(df.isna().any(axis=1).sum())}")
    print(f"  duplicatas exatas     : {int(df.duplicated().sum())}")
    print(f"  num (0–4)             : {df['num'].value_counts().sort_index().to_dict()}")
    pos = int((df["num"] > 0).sum())
    print(f"  binário ausência/pres.: {len(df) - pos}/{pos}  (prevalência {pos / len(df):.1%})")
    for col, dom in CATEGORICAL_DOMAINS.items():
        bad = set(df[col].dropna().astype(int)) - dom
        if bad:
            print(f"  ! {col}: valores fora do domínio {sorted(bad)}")
    print("  domínios categóricos  : verificados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
