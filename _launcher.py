import os
import sys
import subprocess
from pathlib import Path
from typing import List

# Absolute path to silico/ regardless of where the script is called from
SILICO_ROOT = Path(__file__).parent.resolve()

BACKBONES = ["PARENet"]
CKPTS = {
    "PARENet":        SILICO_ROOT / "PARENet"        / "ckpts" / "best.pth.tar",
}



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _experiments_dir_parenet(backbone: str, dataset: str, method: str) -> Path:
    return SILICO_ROOT / backbone / "experiments" / dataset / "tta" / method


def _pythonpath_parenet(backbone: str, dataset: str, method: str) -> str:
    extra = [
        str(SILICO_ROOT / "common"),          # ← add this, makes utils.parser resolvable
        str(SILICO_ROOT / backbone),
        str(SILICO_ROOT / backbone / "experiments" / dataset),
        str(_experiments_dir_parenet(backbone, dataset, method)),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    all_paths = extra + ([existing] if existing else [])
    return os.pathsep.join(all_paths)

def _pythonpath_lepard(backbone: str, dataset: str, method: str) -> str:
    extra = [
        str(SILICO_ROOT / "common"),          # ← add this, makes utils.parser resolvable
        str(SILICO_ROOT / backbone),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    all_paths = extra + ([existing] if existing else [])
    return os.pathsep.join(all_paths)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_script_parenet(backbone: str, dataset: str, method: str, script: str) -> Path:
    path = _experiments_dir_parenet(backbone, dataset, method) / script
    if not path.exists():
        raise FileNotFoundError(
            f"\nScript not found: {path}"
            f"\n  backbone = {backbone}"
            f"\n  dataset  = {dataset}"
            f"\n  method   = {method}"
            f"\n  script   = {script}"
        )
    return path

def launch(backbone: str, dataset: str, method: str, script: str, extra_args: List[str]) -> None:
    script_path = resolve_script_parenet(backbone, dataset, method, script)
    cwd = SILICO_ROOT / backbone / "experiments" / dataset

    env = os.environ.copy()
    env.pop("PYTHONHOME", None)        
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONPATH"] = _pythonpath_parenet(backbone, dataset, method)
    env["TTA_METHOD"]  = method
    env["TTA_DATASET"] = dataset
    

    # forward backbone/dataset/method explicitly so inner parsers receive them
    injected = [
        "--backbone", backbone,
        "--dataset",  dataset,
        "--method",   method,
        #"--snapshot", str(CKPTS[backbone]),  # ← absolute, no relative path
    ]
    cmd = [sys.executable, str(script_path)] + injected + extra_args

    print(f"[launcher] cwd={cwd}")
    print(f"[launcher] {' '.join(cmd)}")
    print(f"[launcher] PYTHONPATH={env['PYTHONPATH']}\n")

    subprocess.run(
        [sys.executable, "-c",
         "import sys; print('[probe] exe', sys.executable);"
         "print('[probe] prefix', sys.prefix);"
         "print('[probe] venv_sp', [p for p in sys.path if 'p2ilreg' in p]);"
         "import torch; print('[probe] torch', torch.__file__)"],
        env=env,
    )
    result = subprocess.run(cmd, env=env, cwd=str(cwd))
    sys.exit(result.returncode)