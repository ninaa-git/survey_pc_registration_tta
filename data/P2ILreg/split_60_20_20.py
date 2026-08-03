import random
from pathlib import Path

DATA_ROOT = Path("/home/kylo-ren/Documents/registration/pc-registration/silico/data/Liver_regis")
SEED = 42
RATIOS = (0.6, 0.2, 0.2)
PATIENTS = [f"{i:02d}" for i in range(1, 22)]

for patient in PATIENTS:
    pdir = DATA_ROOT / patient
    train_f, test_f = pdir / "train_syn.txt.original", pdir / "test_syn.txt.original"

    lines = []
    for f in (train_f, test_f):
        if f.exists():
            lines.extend(f.read_text().splitlines())
    lines = list(dict.fromkeys(l.strip() for l in lines if l.strip()))

    rng = random.Random(SEED + int(patient))   # same seed across patients = fully reproducible
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(n * RATIOS[0])
    n_val   = int(n * RATIOS[1])
    train, val, test = lines[:n_train], lines[n_train:n_train+n_val], lines[n_train+n_val:]

    # backup once
    for f in (train_f, test_f):
        bak = f.with_suffix(".txt.original")
        if f.exists() and not bak.exists():
            f.rename(bak)

    (pdir / "train_syn.txt").write_text("\n".join(train) + "\n")
    (pdir / "val_syn.txt"  ).write_text("\n".join(val)   + "\n")
    (pdir / "test_syn.txt" ).write_text("\n".join(test)  + "\n")

    print(f"P{patient}: total={n}  train={len(train)}  val={len(val)}  test={len(test)}")