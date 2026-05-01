# Evaluation Folder

This folder is for generated benchmark output.

When you run:

```bash
python3 scripts/benchmark_anpr.py path/to/images --labels path/to/labels.csv
```

the script writes:

```text
evaluation/benchmark_predictions.csv
evaluation/benchmark_summary.json
```

Use those numbers to update `RESULTS.md`.

The generated files are ignored by git by default so you do not accidentally
commit personal images/results. If you want to publish final benchmark results,
copy the summary manually into `RESULTS.md`.
