"""Structured, validated study configuration (Step 8A).

A study YAML (configs/*.yaml) declares the fixed operating point plus the
crossed axes; `expand_runs` turns it into the full list of RunSpecs, each with
a deterministic, human-readable run_id and a config hash. Nothing about an
experiment is chosen in source code: every parameter that reaches
`run_mechanism_experiment` comes from the RunSpec, and the RunSpec is written
to the run directory before the run starts.

Hydra (suggested in spec Part 14) is not installed; a plain YAML + explicit
validation layer gives the same composability for this grid without a new
dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.constants import DRIFT_ESTIMATORS, MECHANISMS, METHODS


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    study: str
    run_id: str
    seed: int
    drift_estimator: str
    mechanism: str
    beta: float | None
    method: str
    method_kwargs: dict
    data: dict
    drift: dict
    training: dict
    participation: dict
    condition: str

    def runner_kwargs(self) -> dict:
        t, p = self.training, self.participation
        return dict(
            mechanism=self.mechanism, method=self.method, drift_estimator=self.drift_estimator,
            target_rate=p["target_rate"], gamma=p["gamma"],
            beta=self.beta if self.beta is not None else 0.0,
            threshold_exclude_frac=p.get("threshold_exclude_frac", 0.2),
            model_relative_ema_decay=p.get("model_relative_ema_decay", 0.7),
            rounds=t["rounds"], local_epochs=t["local_epochs"], lr=t["lr"], batch_size=t["batch_size"],
            eval_every=t["eval_every"], seed=self.seed, method_kwargs=dict(self.method_kwargs) or None,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        d = self.to_dict()
        d.pop("run_id")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


REQUIRED = {
    "data": {"n_clients", "n_reserve", "reserve_seed_offset"},
    "drift": {"mode", "fraction", "start_round", "ramp_rounds"},
    "training": {"rounds", "local_epochs", "lr", "batch_size", "eval_every"},
    "participation": {"target_rate", "gamma"},
}


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def load_study(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate_study(cfg)
    return cfg


def validate_study(cfg: dict) -> None:
    _req(isinstance(cfg, dict), "study config must be a mapping")
    for k in ("study", "seeds", "drift_estimators", "conditions", "methods", *REQUIRED):
        _req(k in cfg, f"missing top-level key '{k}'")
    for sec, keys in REQUIRED.items():
        missing = keys - set(cfg[sec])
        _req(not missing, f"section '{sec}' missing keys {sorted(missing)}")
    d, dr, t, p = cfg["data"], cfg["drift"], cfg["training"], cfg["participation"]
    _req(isinstance(d["n_clients"], int) and d["n_clients"] >= 2, "data.n_clients must be an int >= 2")
    _req(dr["mode"] == "label_shift", "drift.mode must be 'label_shift' (rotation is the confounded legacy mode)")
    _req(0 < dr["fraction"] < 1, "drift.fraction must be in (0,1)")
    n_drift = max(1, int(round(d["n_clients"] * dr["fraction"])))
    _req(d["n_reserve"] >= n_drift, f"data.n_reserve must be >= number of drifting clients ({n_drift})")
    _req(1 <= dr["start_round"] <= t["rounds"], "drift.start_round must lie within the run")
    _req(dr["ramp_rounds"] >= 1, "drift.ramp_rounds must be >= 1")
    _req(t["rounds"] >= 1 and t["local_epochs"] >= 1 and t["batch_size"] >= 1 and t["eval_every"] >= 1, "bad training ints")
    _req(t["lr"] > 0, "training.lr must be > 0")
    _req(0.05 <= p["target_rate"] <= 0.95, "participation.target_rate must lie in the positivity band [0.05, 0.95]")
    _req(len(set(cfg["seeds"])) == len(cfg["seeds"]) and all(isinstance(s, int) for s in cfg["seeds"]), "seeds must be unique ints")
    for e in cfg["drift_estimators"]:
        _req(e in DRIFT_ESTIMATORS, f"unknown drift estimator {e}")
    names = set()
    for c in cfg["conditions"]:
        _req(c.get("mechanism") in MECHANISMS, f"unknown mechanism in condition {c}")
        if c["mechanism"] == "mnar":
            _req(isinstance(c.get("beta"), (int, float)) and c["beta"] > 0, f"mnar condition needs beta > 0: {c}")
        else:
            _req("beta" not in c, f"beta only applies to mnar: {c}")
        nm = condition_name(c)
        _req(nm not in names, f"duplicate condition {nm}")
        names.add(nm)
    mnames = set()
    for m in cfg["methods"]:
        _req(m.get("name") in METHODS, f"unknown method {m}")
        _req(m["name"] not in mnames, f"duplicate method {m['name']}")
        mnames.add(m["name"])
        kw = m.get("kwargs", {}) or {}
        if m["name"] == "feddrift":
            _req(set(kw) <= {"delta", "max_models"}, f"feddrift kwargs: {kw}")
            _req(kw.get("delta", 0.04) > 0 and kw.get("max_models", 10) >= 1, "feddrift delta>0, max_models>=1")
        else:
            _req(not kw, f"method {m['name']} takes no kwargs")
    if "expected_total_runs" in cfg:
        total = len(cfg["seeds"]) * len(cfg["drift_estimators"]) * len(cfg["conditions"]) * len(cfg["methods"])
        _req(total == cfg["expected_total_runs"], f"grid has {total} runs but expected_total_runs={cfg['expected_total_runs']}")


def condition_name(c: dict) -> str:
    if c["mechanism"] == "mnar":
        return f"mnar_b{float(c['beta']):g}"
    return c["mechanism"]


def expand_runs(cfg: dict) -> list[RunSpec]:
    """Deterministic order: seed -> estimator -> condition -> method."""
    validate_study(cfg)
    runs = []
    for seed in cfg["seeds"]:
        for est in cfg["drift_estimators"]:
            for c in cfg["conditions"]:
                for m in cfg["methods"]:
                    cond = condition_name(c)
                    est_tag = "mrel" if est == "model_relative" else "mind"
                    runs.append(RunSpec(
                        study=cfg["study"], run_id=f"s{seed}__{est_tag}__{cond}__{m['name']}",
                        seed=int(seed), drift_estimator=est, mechanism=c["mechanism"],
                        beta=float(c["beta"]) if "beta" in c else None, method=m["name"],
                        method_kwargs=dict(m.get("kwargs", {}) or {}), data=dict(cfg["data"]),
                        drift=dict(cfg["drift"]), training=dict(cfg["training"]),
                        participation=dict(cfg["participation"]), condition=cond,
                    ))
    ids = [r.run_id for r in runs]
    _req(len(ids) == len(set(ids)), "run_id collision")
    return runs
