"""Reproduce every thesis number and figure from the raw CSV: python run_all.py"""
import subprocess
import sys
import time

STEPS = [
    ("Phase 1  data loading and validation", "src.data_loader"),
    ("Phase 2  exploratory analysis", "src.eda"),
    ("Phase 3  feature engineering", "src.features"),
    ("Phase 4  chronological splits", "src.splits"),
    ("Phase 4  naive baselines", "src.models.baseline"),
    ("Phase 5  machine learning models", "src.models.ml_models"),
    ("Phase 6  deep learning models", "src.models.dl_models"),
    ("Phase 7  evaluation and comparison", "src.evaluate"),
    ("Phase 8  energy management scenario", "src.scenario"),
    ("Phase 9  appendix tables", "src.appendix"),
]


def main(skip=()):
    t0 = time.perf_counter()
    for title, mod in STEPS:
        if mod in skip:
            print(f"-- skipped {title}"); continue
        print(f"\n== {title} ({mod})")
        s = time.perf_counter()
        subprocess.run([sys.executable, "-m", mod], check=True)   # separate process per step (TF/OpenMP isolation)
        print(f"   done in {time.perf_counter() - s:.1f}s")
    print(f"\nAll steps finished in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(skip=tuple(a for a in sys.argv[1:]))
