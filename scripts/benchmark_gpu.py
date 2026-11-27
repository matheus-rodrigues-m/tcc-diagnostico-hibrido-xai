"""Etapa 0.6 — Benchmark da GPU para as três arquiteturas de Transfer Learning.

Para ResNet-50, DenseNet-121 e EfficientNet-B0 (pesos ImageNet, cabeça de 2 classes):
  1. maior batch que cabe na VRAM em fine-tuning completo (pior caso de memória);
  2. throughput de treino (img/s) com AMP fp16 e modo determinístico;
  3. throughput de inferência (img/s);
  4. estimativa de tempo por época e de horas de GPU do protocolo experimental.

Entrada sintética 224×224 (mede só a GPU; o carregamento de dados será otimizado com cache
de imagens pré-processadas na Fase 4). Resultado: reports/metrics/gpu_benchmark.json

Uso:  uv run python scripts/benchmark_gpu.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime

import torch
import torchvision
from torch import nn

from tcc_diag.paths import METRICS_DIR, MODELS_DIR
from tcc_diag.seed import set_seed

ARCHS = {
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
    "densenet121": (
        torchvision.models.densenet121,
        torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_b0": (
        torchvision.models.efficientnet_b0,
        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}
BATCH_CANDIDATES = [8, 16, 32, 48, 64, 96, 128]
# No Windows, ao esgotar a VRAM o driver NVIDIA pode usar a RAM do sistema ("Sysmem Fallback")
# em vez de dar erro — o treino fica dezenas de vezes mais lento, silenciosamente. Limitamos o
# alocador do PyTorch a 95% da VRAM para que ele mesmo lance OutOfMemoryError antes disso.
VRAM_FRACTION = 0.95
SLOW_STEP_FACTOR = 4.0  # passo >4x mais lento por imagem que o batch anterior => transbordou
# NÃO usar memory_format=channels_last: nesta GPU/versão, channels_last + BatchNorm em modo de
# treino deixou o passo ~9x mais lento (ResNet-50: 23 vs 203 img/s). Medido na etapa 0.6.
IMG = 224
WARMUP, STEPS = 5, 20
INFER_BATCH = 64

# Protocolo previsto (plano mestre, Fases 5.2–5.6) para estimar horas de GPU.
N_TRAIN_IMAGES = 4450  # ~85% de 5.232 imagens do Kermany após split por paciente
N_VAL_IMAGES = 780
EPOCHS = 30  # teto com early stopping (3 épocas de cabeça + até ~27 de fine-tuning)
RUNS = {
    "ablacao_effnet_b0": ("efficientnet_b0", 20),
    "principal_resnet50": ("resnet50", 3),
    "principal_densenet121": ("densenet121", 3),
    "principal_effnet_b0": ("efficientnet_b0", 3),
}


def build(arch: str) -> nn.Module:
    fn, weights = ARCHS[arch]
    model = fn(weights=weights)
    if arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model.cuda()


def train_steps(model: nn.Module, batch: int, steps: int) -> tuple[float, float]:
    """Executa `steps` passos de treino AMP. Retorna (img/s, pico de VRAM em GB)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, IMG, IMG, device="cuda")
    y = torch.randint(0, 2, (batch,), device="cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for i in range(WARMUP + steps):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    assert torch.isfinite(loss), "loss não finita"
    return batch * steps / dt, torch.cuda.max_memory_allocated() / 1024**3


@torch.inference_mode()
def infer_throughput(model: nn.Module, batch: int) -> float:
    model.eval()
    x = torch.randn(batch, 3, IMG, IMG, device="cuda")
    for i in range(WARMUP + STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    return batch * STEPS / (time.perf_counter() - t0)


def max_batch(arch: str) -> tuple[int, dict]:
    """Maior batch de treino sem OOM nem transbordo para a RAM (3 passos por candidato)."""
    tried: dict = {}
    best, prev_ips = 0, None
    for b in BATCH_CANDIDATES:
        model = build(arch)
        try:
            ips, mem = train_steps(model, b, steps=3)
            if prev_ips and ips < prev_ips / SLOW_STEP_FACTOR:
                tried[b] = f"lento ({ips:.0f} img/s): transbordo p/ RAM"
                print(f"    batch {b:>3}: {tried[b]}", flush=True)
                break
            tried[b] = round(mem, 2)
            best, prev_ips = b, ips
            print(f"    batch {b:>3}: OK  pico {mem:.2f} GB  ({ips:.0f} img/s)", flush=True)
        except torch.OutOfMemoryError:
            tried[b] = "OOM"
            print(f"    batch {b:>3}: OOM", flush=True)
            break
        finally:
            del model
            torch.cuda.empty_cache()
    return best, tried


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA indisponível — benchmark requer GPU.")
        return 1
    torch.hub.set_dir(str(MODELS_DIR / "pretrained"))  # pesos no disco do projeto (D:)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    set_seed(42, deterministic=True)
    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {props.name} | VRAM {props.total_memory / 1024**3:.2f} GB | "
        f"torch {torch.__version__}"
    )

    results = {}
    for arch in ARCHS:
        print(f"\n=== {arch} ===")
        n_params = sum(p.numel() for p in build(arch).parameters()) / 1e6
        torch.cuda.empty_cache()
        best, tried = max_batch(arch)
        print(f"  parâmetros: {n_params:.1f} M | VRAM por batch (GB): {tried} -> máx. {best}")
        # throughput medido num batch "de trabalho": 32 se couber, senão o máximo
        work_b = 32 if best >= 32 else best
        model = build(arch)
        ips, mem = train_steps(model, work_b, STEPS)
        inf = infer_throughput(model, INFER_BATCH)
        del model
        torch.cuda.empty_cache()
        epoch_s = N_TRAIN_IMAGES / ips + N_VAL_IMAGES / inf
        print(f"  treino  batch {work_b}: {ips:6.1f} img/s | pico VRAM {mem:.2f} GB")
        print(
            f"  inferência batch {INFER_BATCH}: {inf:6.1f} img/s | "
            f"latência ~{1000 / inf:.1f} ms/img"
        )
        print(f"  época estimada (treino {N_TRAIN_IMAGES} + val {N_VAL_IMAGES}): {epoch_s:5.1f} s")
        results[arch] = {
            "params_M": round(n_params, 2),
            "max_train_batch": best,
            "vram_by_batch_GB": tried,
            "work_batch": work_b,
            "train_img_s": round(ips, 1),
            "train_peak_vram_GB": round(mem, 2),
            "infer_img_s": round(inf, 1),
            "epoch_seconds_est": round(epoch_s, 1),
        }

    print("\n=== Estimativa do protocolo (até 30 épocas por execução; early stopping reduz) ===")
    total_h = 0.0
    plan = {}
    for name, (arch, n_runs) in RUNS.items():
        h = results[arch]["epoch_seconds_est"] * EPOCHS * n_runs / 3600
        plan[name] = {"arch": arch, "runs": n_runs, "hours_upper_bound": round(h, 2)}
        total_h += h
        print(f"  {name:<24} {n_runs:>2} execuções × {arch:<15} ≤ {h:5.2f} h")
    print(f"  TOTAL (limite superior, só GPU): {total_h:.1f} h")

    out = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": props.name,
        "vram_GB": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "settings": {
            "img_size": IMG,
            "vram_fraction_cap": VRAM_FRACTION,
            "amp": "fp16",
            "channels_last": False,
            "deterministic": True,
            "optimizer": "AdamW",
            "full_finetune": True,
            "synthetic_input": True,
        },
        "results": results,
        "protocol_estimate": {
            "epochs_cap": EPOCHS,
            "runs": plan,
            "total_hours_upper_bound": round(total_h, 1),
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "gpu_benchmark.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nSalvo em reports/metrics/gpu_benchmark.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
