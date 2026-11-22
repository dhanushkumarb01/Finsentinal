"""Unit tests for the Step-10 statistics helpers (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import stats as S  # noqa: E402


def test_sign_flip_exact_minimum_p_with_five_pairs():
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert S.sign_flip_p(d) == pytest.approx(2 / 32)          # two-sided floor for n=5
    assert S.sign_flip_p(d, "greater") == pytest.approx(1 / 32)
    assert S.sign_flip_p(-d, "less") == pytest.approx(1 / 32)
    assert S.sign_flip_p(np.zeros(5)) == 1.0


def test_wilcoxon_floor_matches_documented_power_limit():
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert st.wilcoxon(d, method="exact").pvalue == pytest.approx(0.0625)


def test_holm_matches_hand_computation():
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    # sorted: .01*4=.04, .03*3=.09, .04*2=.08->max(.09)=.09, .5*1=.5
    assert S.holm(p).round(6).tolist() == [0.04, 0.09, 0.09, 0.5]


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    x = np.array([0.60, 0.62, 0.61, 0.65, 0.59])
    lo, hi = S.bootstrap_ci(x)
    assert lo <= x.mean() <= hi
    assert S.bootstrap_ci(x) == (lo, hi)


def test_paired_stats_effect_size_and_practical_flag():
    d = np.array([0.02, 0.03, 0.01, 0.025, 0.015])
    r = S.paired_stats(d)
    assert r["d_z"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert r["practically_important"] and r["n_positive"] == 5


def _grid(effect=0.05, seeds=6, noise=0.005):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(seeds):
        seed_off = rng.normal(0, 0.03)
        for c in ["mcar", "mar", "mnar_b2"]:
            for m in ["fedavg", "flfdms"]:
                y = 0.6 + seed_off + (effect if (c == "mnar_b2" and m == "flfdms") else 0) + rng.normal(0, noise)
                rows.append(dict(seed=s, estimator="model_relative", condition=c, mechanism=c.split("_")[0],
                                 beta=2.0 if c.startswith("mnar") else None, method=m, acc_global=y))
    return pd.DataFrame(rows)


def test_mixed_model_detects_planted_interaction():
    r = S.mixed_model(_grid(), "acc_global", "C(condition, Treatment('mcar')) * C(method, Treatment('fedavg'))",
                      {"interaction": "C(condition, Treatment('mcar')) + C(method, Treatment('fedavg'))"})
    term = [c for c in r["coefficients"] if "mnar_b2" in c["term"] and "flfdms" in c["term"]][0]
    assert term["estimate"] == pytest.approx(0.05, abs=0.01)
    assert r["lrt"][0]["p"] < 1e-6


def test_mixed_model_null_interaction_not_significant():
    r = S.mixed_model(_grid(effect=0.0), "acc_global", "C(condition) * C(method)",
                      {"interaction": "C(condition) + C(method)"})
    assert r["lrt"][0]["p"] > 0.01


def test_rm_anova_partial_eta2_in_unit_interval():
    a = S.rm_anova(_grid(), "acc_global")
    assert set(a.effect) == {"condition", "method", "condition:method"}
    assert ((a.partial_eta2 >= 0) & (a.partial_eta2 <= 1)).all()


def test_complete_blocks_drops_incomplete_seed_without_imputing():
    g = _grid()
    g = g[~((g.seed == 3) & (g.method == "flfdms") & (g.condition == "mar"))]
    full, rep = S.complete_blocks(g)
    assert 3 not in set(full.seed) and rep["model_relative"]["seeds_incomplete"] == [3]
    assert len(full) == 5 * 6


def test_did_sign_convention():
    g = _grid(effect=-0.04)   # FL-FDMS loses 4pp under MNAR only
    g.loc[g.method == "flfdms", "method"] = "fedipw_oracle"
    did = S.did_ipw_boundary(g, "acc_global", "fedipw_oracle")
    row = did[(did.ref == "mcar") & (did.condition == "mnar_b2")].iloc[0]
    assert row["mean_diff"] == pytest.approx(-0.04, abs=0.01)


def test_mixed_model_survives_seed_variance_on_boundary():
    """Regression (smoke analysis): with 2 seeds and no seed effect, L-BFGS hit a
    singular Hessian and the whole LMM failed. The fit must fall through to
    another optimizer (or the reported OLS fallback) and still produce LRTs."""
    import numpy as np
    import pandas as pd

    from analysis.stats import mixed_model
    rng = np.random.default_rng(0)
    rows = [dict(seed=s, condition=c, method=m, acc=0.5 + 0.02 * (m == "b") + rng.normal(0, 0.01))
            for s in (0, 1) for c in ("mcar", "mnar") for m in ("a", "b", "c")]
    d = pd.DataFrame(rows)
    out = mixed_model(d, "acc", "C(condition) * C(method)", {"interaction": "C(condition) + C(method)"})
    assert "fit_method" in out and out["coefficients"]
    assert out["lrt"][0]["df"] == 2
