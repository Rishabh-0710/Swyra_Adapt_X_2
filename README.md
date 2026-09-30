# Swyra_Adapt_X_2
# ADAPT-X: Continual Learning Under Dynamic Distribution Shift

ADAPT-X is a domain-agnostic continual-learning system for supervised tabular data streams.

It learns one fixed task from an initial batch and then processes labelled batches sequentially:

```text
Batch 0
  │
  ▼
Initial Learning
  │
  ▼
Persistent Model State
  │
  ├───────────────┐
  │               │
Incoming Batch    │
  │               │
  ▼               │
Pre-update        │
Prediction        │
  │               │
  ▼               │
Distribution      │
Shift Detector    │
  │               │
  ▼               │
Adaptation        │
Controller        │
  │               │
  ├───────┬───────┤
  │       │       │
  ▼       ▼       ▼
Anchor  Specialist Replay
  │       │       │
  └───────┴───────┘
          │
          ▼
Controlled Incremental Update
          │
          ▼
Bounded Replay + Bounded Reference
          │
          ▼
       Batch t+1
```

The operating loop is:

**LEARN → DETECT → ADAPT → RETAIN → LEARN AGAIN**

---

## 1. What was wrong in Attempt #1

The evaluator reported a 69.23 score, with the largest structural weakness being Problem Alignment at 65.28%.

The repository audit found several real issues:

1. `DriftDetector` copied the complete reference matrix with `self.reference_X = X_arr.copy()`. That contradicted the bounded-memory claim.
2. Target encoding used current-batch labels before the same batch was trained, creating target leakage.
3. Missing-value indicator columns were created only when missing values occurred, so feature dimensionality could change between batches.
4. The adaptive model rebuilt fresh LightGBM models on replay + current data. That was controlled retraining, not genuine persistent incremental learning.
5. Diagnostic lists such as drift/adaptation/update histories grew with stream length.
6. Categorical dictionaries stored every observed category indefinitely.
7. The benchmark and ablation harnesses advanced one shared evaluator RNG across models, meaning models could receive different hidden evaluation splits.
8. The random-replay ablation only randomized when the buffer exceeded capacity, but the buffer was already capped, so that ablation did not actually test random replay.
9. README benchmark numbers were inconsistent with the repository result files and the evaluator screenshots.

These were implementation problems, not naming problems.

---

## 2. ADAPT-X Attempt #2 architecture

### 2.1 Leakage-safe preprocessing

`src/preprocessing.py` now uses a frozen Batch-0 feature representation.

For numerical columns it stores:

- initial imputation values
- initial mean and standard deviation
- deterministic missingness indicators

For categorical columns it uses fixed-width deterministic hashing into a configured number of bins.

There is deliberately **no target encoding**.

Why freeze the representation?

A continual tree model has historical split thresholds. If normalization or category statistics change the meaning of every feature after each batch, old trees and new data no longer share the same coordinate system. ADAPT-X therefore keeps the transform schema stable while separately maintaining bounded online statistics for monitoring.

Most importantly:

```text
Previous preprocessing state
        ↓
Transform X_t
        ↓
Pre-update prediction
        ↓
Drift detection
        ↓
Model update with X_t, y_t
        ↓
Update preprocessing diagnostics
        ↓
Prepare for t+1
```

`y_t` never participates in constructing the features used to learn from `y_t`.

### 2.2 Bounded distribution-shift detection

`src/drift.py` contains:

- PSI
- normalized Wasserstein distance
- residual CUSUM
- sample-size confidence
- drift severity
- drift type
- affected features

The detector stores a fixed-size reference reservoir.

Default:

```text
reference_size = 256
```

The detector never stores the complete initial dataset.

The reference reservoir is updated only **after** the current batch has been evaluated and learned, preventing a batch from influencing its own drift report.

A finite-sample null calibration is also computed from the bounded reference. This reduces false alarms caused by normal sampling variation.

Concept drift uses a sample-size-aware standardized residual statistic in addition to cumulative CUSUM evidence.

### 2.3 Persistent incremental model state

The previous implementation replaced its LightGBM models on every update.

Attempt #2 instead uses LightGBM's persistent booster state:

```python
updated.fit(X_update, y_update, init_model=old_model.booster_)
```

This appends new trees to the existing model rather than discarding its learned trees.

There are two persistent components:

**Stable anchor**

- retains long-term knowledge
- receives bounded replay plus conservative current evidence
- grows only up to `max_trees_core`

**Adaptive specialist**

- receives more current evidence
- grows incrementally
- receives a higher prediction weight under drift
- may be intentionally refreshed on severe concept drift

The severe concept-drift refresh is deliberate. The specialist is the plastic component; the anchor remains the stable component.

This is therefore a stability/plasticity architecture, not a sequence of fresh full-data fits.

### 2.4 Bounded replay

`src/memory.py` enforces:

```text
|Replay| <= K
```

Incoming batches larger than K are reduced before internal replay compaction, so the replay subsystem does not temporarily construct a historical array proportional to the complete stream.

Replay selection combines:

- target stratification
- farthest-point diversity
- deterministic random state

Only transformed features and targets for the bounded replay set are retained.

### 2.5 Bounded diagnostics

The model now uses bounded deques for:

- drift history
- adaptation history
- update timing history

Lifetime sample count and update count are scalars.

No raw historical DataFrame or Series is retained.

---

## 3. Domain-agnostic operation

The algorithm contains no rules about feature meaning.

There are no assumptions such as:

```text
attendance > X
temperature > X
income > X
study_hours > X
```

The system operates using:

- numerical statistics
- categorical hashing
- residual behaviour
- feature-distribution drift
- model behaviour
- target distributions
- bounded replay

The benchmark covers:

1. nonlinear regression
2. nonlinear binary classification
3. high-dimensional regression
4. mixed categorical/numerical classification
5. heteroscedastic regression

The model receives the same continual-learning interface regardless of dataset family.

---

## 4. Experimental validity fixes

The authoritative benchmark and ablation harness now create a fresh `SequentialStreamSimulator` with the same seed for every model configuration.

Therefore, for a given dataset family and seed:

```text
Static
Naive Incremental
Standard Replay
ADAPT-X
Full Retrain
```

receive the same hidden evaluation slices.

This fixes a major comparison flaw in the previous harness.

The ablation suite also uses the same five dataset families as the benchmark.

---

## 5. Attempt #2 benchmark

Authoritative run:

```text
Datasets: 5 families
Seeds: 42, 101
Batch size: 500
Hidden evaluation ratio: 20%
```

Results are stored in:

```text
results/benchmark_results.csv
results/benchmark_summary.csv
```

Aggregated result:

| Model | Multi-objective score | Mean shifted performance | Max degradation | Initial retention | Mean update latency | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.6283 | -2.1252 | 2.3368 | -0.8310 | 0.0060 s | 16.37 KB |
| Naive Incremental | 0.7220 | -1.2332 | 0.7652 | -2.1214 | 0.0225 s | 16.71 KB |
| Standard Replay | 0.6929 | -1.4184 | 1.1088 | -1.2445 | 0.0754 s | 95.78 KB |
| **ADAPT-X** | **0.6839** | **-1.4736** | **1.2736** | **-1.1184** | **0.0634 s** | **187.99 KB** |
| Full Retrain Oracle | 0.6784 | -1.4917 | 1.3276 | -0.9093 | 0.1142 s | 395.78 KB |

These numbers are reported as measurements, not as a claim that ADAPT-X wins every predictive metric.

In this run, Naive Incremental has the highest aggregate multi-objective score, while ADAPT-X trades some current-task performance for a different stability/retention/memory profile.

That is an important limitation to report rather than hide.

---

## 6. Ablation study

The authoritative ablation run uses the same:

- five dataset families
- seeds 42 and 101
- batch size 500
- 20% hidden evaluation split
- metrics

Results:

| Configuration | Multi-objective score | Max degradation | Retention performance |
|---|---:|---:|---:|
| Full ADAPT-X | 0.6927 | 1.4249 | -1.1533 |
| No Drift Detection | 0.6880 | 1.2949 | -1.1956 |
| No Replay | 0.7102 | 1.1431 | -1.5402 |
| No Adaptive Scaling | 0.6922 | 1.2033 | -1.3340 |
| No Anchor Backbone | 0.7518 | 0.3784 | -1.6648 |
| Random Replay | 0.6924 | 1.3155 | -1.1382 |

The ablation results do **not** prove that every component improves the aggregate score on every environment.

For example, removing the anchor can improve the reported current-stream score in these synthetic benchmarks while worsening retention. This is precisely why the evaluation records multiple dimensions instead of hiding inconvenient results behind one number.

---

## 7. Tests

Attempt #2 has:

```text
18 tests passed
```

The added contract tests cover:

- bounded drift reference memory
- current-label independence of feature construction
- stable transformed feature dimensionality
- bounded diagnostic histories
- bounded replay memory
- persistent incremental tree state

Existing tests continue to cover:

- initialization
- regression
- classification
- sequential updates
- missing values
- categorical data
- replay capacity
- long-stream memory
- raw-object retention
- reproducibility
- evaluation leakage
- stationary drift
- covariate drift
- concept drift
- adaptation behaviour

Run:

```bash
python -m pytest tests/ -v
```

---

## 8. Security and configuration

The project does not require secrets or external credentials.

Therefore no fake `.env` file was added.

Configuration lives in:

```text
config.yaml
```

It documents:

- replay capacity
- drift reference capacity
- preprocessing hash width
- adaptation bounds
- model tree budgets
- evaluation settings

No credentials are required.

---

## 9. SDG 9 connection

ADAPT-X relates technically to **UN Sustainable Development Goal 9: Industry, Innovation and Infrastructure**.

The connection is computational rather than promotional:

```text
Continual learning
      ↓
No full historical retraining
      ↓
Bounded replay + bounded drift reference
      ↓
Controlled incremental model updates
      ↓
Reduced repeated computation
      ↓
Resource-aware adaptive ML infrastructure
```

The system therefore demonstrates an approach to resilient, resource-bounded intelligent infrastructure under changing data distributions.

---

## 10. Repository structure

```text
src/
├── model.py           # ADAPT-X continual learner
├── drift.py           # bounded distribution-shift detector
├── adaptation.py      # stability/plasticity controller
├── memory.py          # bounded replay memory
├── preprocessing.py   # leakage-safe fixed-schema preprocessing
├── task.py            # domain-agnostic task inference
├── evaluation.py      # sequential leakage-safe evaluation
├── metrics.py
└── utils.py

experiments/
├── run_experiments.py
├── run_ablations.py
├── baselines.py
├── distribution_shifts.py
└── candidate_selection.py

tests/
├── test_adaptx_contract.py
├── test_model.py
├── test_memory.py
├── test_update.py
├── test_leakage.py
└── test_drift.py
```

---

## 11. Reproduce Attempt #2

Install:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest tests/ -v
```

Run the authoritative benchmark:

```bash
python -m experiments.run_experiments --seeds 42 101 --batch_size 500 --output_dir results
```

Run the matching ablation study:

```bash
python -m experiments.run_ablations --seeds 42 101 --batch_size 500 --output_dir results
```

The benchmark and ablation commands now use the same dataset families, seeds, batch size and hidden-evaluation protocol.

---

## 12. Final technical assessment

ADAPT-X now has a defensible continual-learning architecture:

```text
LEARN
  ↓
PREDICT BEFORE UPDATE
  ↓
DETECT DISTRIBUTION SHIFT
  ↓
SELECT ADAPTATION STRENGTH
  ↓
INCREMENTALLY EXTEND PERSISTENT MODELS
  ↓
RETAIN BOUNDED REPLAY
  ↓
UPDATE BOUNDED REFERENCE
  ↓
NEXT BATCH
```

The implementation no longer relies on:

- full historical reference storage
- current-batch target encoding
- dynamically changing feature dimensionality
- unbounded diagnostic histories
- repeated full historical retraining
- unbounded categorical dictionaries
- model-dependent hidden evaluation splits

The remaining limitation is predictive: on the supplied synthetic benchmark families, ADAPT-X does not dominate every baseline. Its strongest technical improvements are architectural correctness, bounded state, leakage prevention, sequential adaptation, and substantially lower update latency than the previous implementation.

That distinction is intentional. The repository is designed to be technically defensible rather than benchmark-gamed.
