"""Import first in every entry script: keeps all caches inside the project."""
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for k, v in {"HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch", "XDG_CACHE_HOME": ".cache",
             "MPLCONFIGDIR": ".cache/mpl"}.items():
    os.environ.setdefault(k, os.path.join(_root, v))
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
