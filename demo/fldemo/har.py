"""UCI HAR loading, fixed subject split, and the documented 64-feature subset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from fldemo.paths import FEATURES_FILE, HAR_ROOT, HAR_ZIP, HAR_ZIP_SHA256, SPLIT_FILE

NUM_CLASSES = 6
NUM_FEATURES = 64
ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


@dataclass(frozen=True)
class HarData:
    x: np.ndarray        # (N, 561) float32, already scaled to [-1, 1] by the dataset authors
    y: np.ndarray        # (N,) int64, 0..5
    subject: np.ndarray  # (N,) int64, 1..30
    feature_names: tuple[str, ...]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive() -> bool:
    return HAR_ZIP.exists() and sha256_file(HAR_ZIP) == HAR_ZIP_SHA256


@lru_cache(maxsize=1)
def load_har() -> HarData:
    """Load train+test parts of UCI HAR into one subject-indexed table."""
    if not HAR_ROOT.exists():
        raise FileNotFoundError(f"UCI HAR not found at {HAR_ROOT}. Run demo/setup.py first.")
    xs, ys, ss = [], [], []
    for part in ("train", "test"):
        d = HAR_ROOT / part
        xs.append(np.loadtxt(d / f"X_{part}.txt", dtype=np.float32))
        ys.append(np.loadtxt(d / f"y_{part}.txt", dtype=np.int64) - 1)
        ss.append(np.loadtxt(d / f"subject_{part}.txt", dtype=np.int64))
    names = tuple(
        line.split(maxsplit=1)[1].strip()
        for line in (HAR_ROOT / "features.txt").read_text().splitlines()
        if line.strip()
    )
    return HarData(np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), names)


def load_split() -> dict:
    return json.loads(SPLIT_FILE.read_text())


def subjects_mask(data: HarData, subjects) -> np.ndarray:
    return np.isin(data.subject, np.asarray(list(subjects)))


def select_features(
    x: np.ndarray, y: np.ndarray, k: int = NUM_FEATURES, max_abs_corr: float = 0.90
) -> list[int]:
    """Greedy ANOVA-F ranking with redundancy pruning.

    Features are ranked by their one-way ANOVA F statistic across the six
    activities; a feature is skipped if its |Pearson r| with an already chosen
    feature exceeds ``max_abs_corr``. Deterministic for fixed inputs.
    """
    from sklearn.feature_selection import f_classif

    f, _ = f_classif(x, y)
    f = np.nan_to_num(f, nan=0.0)
    order = np.argsort(-f, kind="stable")
    xc = x - x.mean(axis=0)
    norms = np.linalg.norm(xc, axis=0) + 1e-12
    chosen: list[int] = []
    for j in order:
        if len(chosen) == k:
            break
        if chosen:
            r = (xc[:, chosen].T @ xc[:, j]) / (norms[chosen] * norms[j])
            if np.max(np.abs(r)) > max_abs_corr:
                continue
        chosen.append(int(j))
    if len(chosen) < k:
        raise RuntimeError(f"Only {len(chosen)} features passed pruning; relax max_abs_corr.")
    return chosen


@lru_cache(maxsize=1)
def feature_indices() -> tuple[int, ...]:
    spec = json.loads(FEATURES_FILE.read_text())
    return tuple(spec["indices"])


def subject_arrays(subject_ids) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[:, subset], y) for the given subjects, in file order."""
    data = load_har()
    m = subjects_mask(data, subject_ids)
    idx = list(feature_indices())
    return data.x[m][:, idx].copy(), data.y[m].copy()
