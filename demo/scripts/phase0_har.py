"""Phase 0: verify the HAR archive, choose the feature subset, train the MLP centrally.

Usage:  demo/.venv/Scripts/python demo/scripts/phase0_har.py [--select]

--select  (re)computes the feature subset from the 'selection' subjects only and
          writes demo/config/har_features.json. Without it, the committed file is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fldemo import MODEL_VERSION  # noqa: E402
from fldemo.har import (  # noqa: E402
    NUM_FEATURES, load_har, load_split, select_features, subject_arrays, subjects_mask, verify_archive,
)
from fldemo.model import PARAM_COUNT, accuracy, new_model, train_sgd  # noqa: E402
from fldemo.paths import FEATURES_FILE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()

    print("archive sha256 ok:", verify_archive())
    data = load_har()
    split = load_split()
    print("samples:", len(data.y), "subjects:", len(np.unique(data.subject)))

    if args.select or not FEATURES_FILE.exists():
        m = subjects_mask(data, split["selection"])
        idx = select_features(data.x[m], data.y[m], k=NUM_FEATURES, max_abs_corr=0.90)
        FEATURES_FILE.write_text(json.dumps({
            "_doc": "64 of the 561 UCI HAR features, chosen ONLY on the 'selection' subjects of "
                    "har_split.json: greedy one-way ANOVA-F ranking, skipping any feature with "
                    "|Pearson r| > 0.90 to an already chosen one (fldemo.har.select_features). "
                    "Order matters: it is the model's input order on every device.",
            "method": "anova_f_greedy_corr_prune", "max_abs_corr": 0.90,
            "selection_subjects": split["selection"],
            "indices": idx,
            "names": [data.feature_names[i] for i in idx],
        }, indent=1))
        print("wrote", FEATURES_FILE)

    torch.manual_seed(0)
    torch.set_num_threads(1)  # stay light: the research sweep owns the CPU
    device_subjects = list(split["software_devices"].values()) + list(split["wokwi_boards"].values())
    train_subjects = device_subjects + split["selection"] + split["drift_source"]
    xtr, ytr = subject_arrays(train_subjects)
    xte, yte = subject_arrays(split["test"])
    model = new_model(seed=0)
    print(f"model {MODEL_VERSION}: params={PARAM_COUNT}  train={len(ytr)} test={len(yte)}")
    print(f"  untrained test acc: {accuracy(model, xte, yte):.4f}")
    losses = train_sgd(model, xtr, ytr, epochs=30, lr=0.05, batch_size=32, seed=0)
    print(f"  central SGD 30 epochs: final loss {losses[-1]:.4f}  test acc {accuracy(model, xte, yte):.4f}")

    # Per-device local accuracy of the central model (sanity check for non-IID spread).
    for name, subj in {**split["software_devices"], **split["wokwi_boards"]}.items():
        x, y = subject_arrays([subj])
        counts = np.bincount(y, minlength=6).tolist()
        print(f"  {name} subject {subj:2d}: n={len(y)} classes={counts} acc={accuracy(model, x, y):.3f}")


if __name__ == "__main__":
    main()
