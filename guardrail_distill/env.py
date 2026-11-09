"""Import first in every entry script: project-local caches, CPU-thread cap, and per-phase paths."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for k, v in {"HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch", "XDG_CACHE_HOME": ".cache",
             "MPLCONFIGDIR": ".cache/mpl", "TRITON_CACHE_DIR": ".cache/triton"}.items():
    os.environ.setdefault(k, os.path.join(ROOT, v))
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# CPU cap: the job scheduler gives each process a share of the 21-core budget via GD_THREADS.
_threads = os.environ.setdefault("GD_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, _threads)

PHASES = {"phase1": "rental_support_v1", "phase2": "rental_support"}  # phase -> domain


def paths(phase):
    """data/<phase>, results/<phase>, ckpt/<phase> (relative to the repo root)."""
    return {k: os.path.join(ROOT, k, phase) for k in ("data", "results", "ckpt")}


def set_torch_threads():
    import torch
    torch.set_num_threads(int(os.environ["GD_THREADS"]))
