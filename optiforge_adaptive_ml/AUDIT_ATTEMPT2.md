# OPTIFORGE Attempt #2 Technical Audit

## Attempt #1 findings

- DriftDetector retained the complete initial reference matrix.
- Target encoding used labels from the current batch before training on that batch.
- Missing-value indicators were dynamically added.
- Anchor and specialist models were rebuilt from replay/current data on every update.
- Diagnostic histories grew with stream length.
- Categorical dictionaries could grow with the number of observed categories.
- Benchmark models did not share identical hidden evaluation splits because one evaluator RNG was reused across model runs.
- Random replay ablation was ineffective because the buffer was already capped before its overflow-only randomization branch.
- README and result artifacts contained inconsistent benchmark values.

## Attempt #2 changes

### Memory
- Drift reference is a fixed reservoir (`reference_size=256` by default).
- Replay is hard capped (`K=500` by default).
- Large incoming batches are reduced to a bounded candidate set before replay compaction.
- Categorical state is fixed-width hashed arrays rather than unbounded dictionaries.
- Diagnostic histories are bounded deques.
- LightGBM tree counts are capped for both persistent components.

### Leakage
- Removed target encoding from the feature path.
- Current batch is transformed before its labels are incorporated into any state.
- `partial_fit` only updates bounded diagnostics and never changes transform parameters.
- Drift reference is updated only after current-batch evaluation and learning.
- Hidden evaluation samples remain outside model calls.

### Continual learning
- Anchor and specialist use LightGBM `init_model` to extend persistent booster state.
- Severe concept drift can refresh only the adaptive specialist.
- The anchor remains the stable long-term component.
- Replay is bounded and never reconstructed from historical batches.

### Evaluation
- Every model receives the same hidden split for a given dataset/seed.
- Ablations and benchmarks use the same five dataset families.
- Random replay ablation now actually randomizes the bounded buffer.

## Attempt #2 authoritative run

- Seeds: 42, 101
- Batch size: 500
- Dataset families: 5
- Hidden evaluation ratio: 20%
- Tests: 18 passed

See `results/benchmark_summary.csv` and `results/ablation_summary.csv` for measured results.

## Limitations

The measured predictive benchmark does not show ADAPT-X dominating every baseline. In particular, the supplied synthetic environments can reward simple batch-specific adaptation. The implementation therefore does not claim universal predictive superiority. The main improvements are architectural correctness, leakage prevention, bounded state, reproducibility, and efficient sequential updates.
