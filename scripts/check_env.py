"""Etapa 0.1 — Verificação do ambiente.

Confere a versão do Python, importa todas as bibliotecas do projeto, imprime as versões
e executa uma operação real na GPU com precisão mista (AMP) para garantir que os
kernels CUDA funcionam, não apenas que a GPU é "visível".

Uso:  uv run python scripts/check_env.py
Saída: código 0 se tudo passou; 1 caso contrário.
"""

from __future__ import annotations

import importlib
import platform
import sys

# (nome do módulo para import, rótulo exibido)
MODULES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("pyarrow", "pyarrow"),
    ("yaml", "pyyaml"), ("joblib", "joblib"),
    ("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost"),
    ("optuna", "optuna"), ("shap", "shap"), ("lime", "lime"), ("statsmodels", "statsmodels"),
    ("scikit_posthocs", "scikit-posthocs"),
    ("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv"),
    ("pydicom", "pydicom"), ("PIL", "pillow"), ("imagehash", "imagehash"),
    ("pytorch_grad_cam", "grad-cam"),
    ("streamlit", "streamlit"), ("pydantic", "pydantic"),
    ("pytest", "pytest"), ("tcc_diag", "tcc_diag (pacote do projeto)"),
]


def main() -> int:
    failures: list[str] = []

    print(f"Python  {platform.python_version()}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python esperado 3.12, encontrado {platform.python_version()}")

    print("\n--- Bibliotecas ---")
    for mod_name, label in MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "?")
            print(f"  OK   {label:<28} {version}")
        except Exception as exc:  # noqa: BLE001 — queremos relatar qualquer falha de import
            print(f"  FALHA {label:<27} {type(exc).__name__}: {exc}")
            failures.append(f"import {label}")

    print("\n--- GPU / CUDA ---")
    import torch

    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() == False")
        print("  CUDA indisponível")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"  Dispositivo : {props.name}")
        print(f"  VRAM        : {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute cap.: {props.major}.{props.minor}")
        print(f"  CUDA (torch): {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
        # Operação real: convolução + matmul em fp16 via autocast (AMP), como no treino.
        x = torch.randn(8, 3, 224, 224, device="cuda")
        conv = torch.nn.Conv2d(3, 16, 3).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = conv(x).flatten(1) @ torch.randn(16 * 222 * 222, 10, device="cuda")
        torch.cuda.synchronize()
        assert y.shape == (8, 10) and torch.isfinite(y).all(), "saída inválida na GPU"
        print(f"  Teste AMP   : OK (saída {tuple(y.shape)}, dtype {y.dtype})")

    print("\n" + ("=" * 50))
    if failures:
        print("RESULTADO: FALHOU ->", "; ".join(failures))
        return 1
    print("RESULTADO: TODAS AS VERIFICAÇÕES PASSARAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
