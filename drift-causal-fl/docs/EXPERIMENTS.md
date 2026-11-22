# Experiment infrastructure and study protocol (Steps 8–9)

## 1. Configuration (`configs/*.yaml`, `src/experiments/config.py`)

A study is one YAML file. It holds the fixed operating point and the crossed
axes: seeds × drift estimators × conditions × methods. `load_study` validates
it before anything runs, and invalid files are rejected with a `ConfigError`.
The validator checks:

- required sections and keys are present
- mechanism, method and estimator names exist
- β is given for MNAR only, and is > 0
- no condition, method or seed appears twice
- the target rate lies in the positivity band
- the reserve pool covers every drifting client
- the drift mode is `label_shift`
- method kwargs are valid
- the expanded grid matches `expected_total_runs`

`expand_runs` produces one frozen `RunSpec` per cell. Each has a readable
`run_id`, e.g. `s3__mind__mnar_b4__feddrift`, and a SHA-256 config hash.
Every argument the simulator receives comes from the RunSpec; experiment
conditions are never set by editing source code.

| File | Purpose | Runs |
|---|---|---|
| `configs/mvp.yaml` | The spec Part 8 MVP matrix | **350** = 7 conditions × 5 methods × 5 seeds × 2 drift definitions |
| `configs/smoke.yaml` | Spec Day-7 reduced smoke sweep: 20 clients, 15 rounds, 5 conditions, 5 methods, 2 seeds, 2 estimators | 100 |
| `configs/pilot_calibration.yaml` | Held-out tuning seed 100: lr stability check and FedDrift δ calibration | 1 |

### Operating point (MVP)

| Axis | Value | Source |
|---|---|---|
| Data | FEMNIST, 100 writer-partitioned clients per seed (80/20 split per writer) | spec |
| Drift | label-shift: ~30% of clients (30) progressively take over a distinct reserve writer's data | spec Part 3 / Step-5 confound fix |
| Drift timing | onset at round 20, linear ramp to full severity over 10 rounds | chosen |
| Rounds / local epochs / eval | 100 / 1 / every 5 rounds | spec |
| Optimiser | SGD, lr 0.03, momentum 0.9, batch 10 | Day-5 operating point; LEAF batch size |
| Dropout | matched marginal rate p = 0.30; MCAR, MAR (γ = 2), MNAR-logistic β ∈ {0.5, 1, 2, 4, 8} | spec |
| Seeds | 0–4. One seed fixes writer sample, reserve pool (seed + 10000), drift cohort and sources, model init, and mechanism RNG | spec Part 14 |

**Paired design.** Within a seed, every condition and method gets the identical
clients, drift schedule and evaluation sets. The seed is therefore the
experimental unit for all paired statistics.

**Deterministic duplicates.** MCAR and MAR never read D, so each
model-independent MCAR/MAR run replicates its model-relative twin bit for bit.
That is 50 runs (2 conditions × 5 methods × 5 seeds). They still run, as the
spec matrix requires, but are scheduled last (`--order unique_first`). The
analysis then uses them as an end-to-end reproducibility check
(`reproducibility_mcar_mar_duplicates.csv`).

## 2. Logging and traceability

Each run gets its own directory, `results/<study>/<run_id>/`:

| File | Contents |
|---|---|
| `spec.json` | RunSpec, config hash, environment (Python/torch/numpy versions, OS, CPU, threads, flush-denormal flag, git commit + dirty flag), data provenance (cache file, SHA-256 of client and reserve tensors, drift cohort, drift-source map) |
| `metrics.jsonl` | Appended incrementally at every evaluation round: global, drift-cohort, stable-cohort, worst-client and clean accuracy, across-client SD, participants, expected/realized dropout, realized ρ (estimator D and true D), estimator fidelity corr(D, D_true), drifting participants, elapsed time, and the method diagnostics (see below) |
| `trace.npz` | Every round × every client: participation mask R, true miss-probability p_miss, D_true, D_model, D_indep |
| `result.json` | Final result: config, eval-round log, summary, per-client final rows, final method state summary |
| `attempts.jsonl` | One line per attempt: start/end, status, duration, RSS, error, resumed-from-checkpoint |
| `error.txt` | Traceback of the last failed attempt |

Method diagnostics written to `metrics.jsonl`:

- **FedIPW:** per-participant propensities, weight ratio, effective sample size
- **FL-FDMS:** friend map, number substituted or without a friend, same-cohort friend fraction
- **FedDrift:** number of models, cluster sizes, flagged clients, merges and distances, assignment vector, drift-test statistic

## 3. Checkpointing, recovery, duplicate prevention

- A checkpoint is written every 10 rounds (atomic write). It holds the full
  method state, both RNG states, both drift trackers, the lagged model(s) and
  all logs. Resuming is bit-identical to an uninterrupted run (tested for every
  method, at both the runner and the sweep layer). The checkpoint is deleted
  once `result.json` exists.
- A worker claims a run by creating its `lock` file with O_EXCL. A lock whose
  pid is dead on this host is stale and gets reclaimed.
- A completed run is never executed again.
- A failed run is never retried silently. Rerunning one requires
  `--retry-failed` after the cause has been fixed.
- An existing `spec.json` whose config hash differs from the current spec
  refuses to run, so results from different configs never mix.

## 4. Commands

```bash
cd drift-causal-fl
python scripts/prepare_data.py --config configs/mvp.yaml                  # build per-seed tensor caches (once)
python scripts/run_sweep.py --config configs/mvp.yaml --dry-run           # validate + show plan
python scripts/run_sweep.py --config configs/mvp.yaml --workers 4 --threads 1
python scripts/run_sweep.py --config configs/mvp.yaml --status            # done / running / failed / pending
python scripts/run_analysis.py --study mvp                                # Step 10 (works on partial results too)
python -m pytest tests -q
```

The sweep can be killed and restarted at any time. It also runs on any other
machine: copy the repository and `data_cache/`, run the same command, and
merge the run directories.

## 5. Resource measurements (this machine)

Hardware: i5-12450H (4P + 4E cores, 12 threads), 15.7 GB RAM, no GPU.

- **Per-round cost** at 100 clients (≈12.4 k training samples): local training
  dominates. The model-relative D forward pass over all 17.7 k samples costs
  ~15%; model initialisation ~2%.
- **SGD:** `foreach=True` is bit-identical to the default per-parameter
  update and ~32% faster, so it is used everywhere.
- **Throughput:** many single-thread workers beat one multi-thread process
  (6×1 thread: 563 samples/s; 4×2: 511; 1×8: 200; 3×3: badly oversubscribed).
- **Memory is the binding constraint on a shared desktop.** A worker uses
  1.0–1.4 GB; FL-FDMS holds up to |S_t| × 6.6 M float32 update vectors
  (~1.8 GB at 70 participants). With other applications open, more than 4
  workers pushed the machine into paging.
- **Flush-to-zero for subnormal floats** is enabled in sweep workers and
  recorded in `spec.json`.

## 6. FedDrift δ

δ = 0.04 was selected on the held-out tuning seed 100 using the paper grid
{0.02, …, 0.20} and maximum Youden J: TPR 0.244, FPR 0.096, J 0.149. The full
grid is in `results/pilot_calibration/delta_calibration.json`, and the
discussion is in the "FedDrift δ calibration" section of PROGRESS.md. The same
pilot confirmed that lr = 0.03 is stable over 100 rounds (final accuracy 0.825).
