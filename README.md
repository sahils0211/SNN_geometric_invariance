# SNN Geometric Invariance

Research code for studying rotation invariance in spiking neural networks (SNNs) using p4-equivariant group CNNs. This project targets an ICLR-style investigation of how architectural equivariance and spiking configurations affect accuracy, rotation robustness, temporal behavior, and internal representations.

The work implements the research plan of Prof. K. V. Subrahmanyam.

## Models

Two p4-equivariant SNN architectures are included:

- **Full-capacity model** — a 7-layer, 10-channel model with **24,744 parameters**.
  Main files: `gconv.py`, `p4cnn.py`, `spiking_p4cnn.py`, and `train_spiking.py`.
- **Reduced-capacity model** — a 5-layer, 4-channel ablation with **2,420 parameters**.
  Main files: `gconv_small.py`, `p4cnn_small.py`, `spiking_p4cnn_small.py`, and `train_spiking_small.py`.

Both configurations compare five modes:

- `none` — analog control
- `alternating`
- `full`
- `hybrid_early`
- `hybrid_late`

Experiments use MNIST digits 3, 4, 8, and 9, evaluated across rotations from 0° to 345° in 15° increments.

## Setup

Create and activate a Python environment, then install the packages used by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy snntorch pandas matplotlib
```

Download the Kaggle Digit Recognizer training data and place it at:

```text
~/Downloads/digit-recognizer/train.csv
```

The CSV must use the standard Kaggle layout: a `label` column followed by 784 pixel columns.

## Reproducing the reduced-capacity experiments

From the repository root:

```bash
source .venv/bin/activate
./run_small_sweep.sh
```

This runs the five modes over seeds 0, 1, and 2 using the reduced-capacity model at `T=50`, then aggregates per-mode results and creates the cross-mode comparison. It uses GPU 0 by default.

To reproduce the `T=30` version on GPU 1:

```bash
source .venv/bin/activate
./run_small_sweep_gpu1_t30.sh
```

The training script can also be run for an individual configuration:

```bash
python train_spiking_small.py --mode full --seed 0
```

After all seeds for a mode have completed, aggregate results and compare modes:

```bash
python train_spiking_small.py --aggregate --mode full
python train_spiking_small.py --compare --modes none alternating full hybrid_early hybrid_late
```

Training writes checkpoints, JSON results, and plots into ignored output directories (`models_small*/`, `results_small*/`, and `plots_small*/`).

## Representation analyses

After training checkpoints exist, generate the reduced-model representation figures:

```bash
./run_all_representation_plots.sh
```

This produces rotation-sensitivity, direct-equivariance-error, representation-similarity (cosine/CKA/SVCCA), spike-accumulation, and truncated-time accuracy plots.

To run both available checkpoint conditions:

```bash
./run_repr_analysis_both.sh
```

This analyzes the T=50 checkpoints in `models_small/` and T=30 checkpoints in `models_small_t30/`.

For complete multi-seed analyses at T=50:

```bash
./run_repr_analysis_multiseed.sh
```

For T=30 checkpoints, specify the alternate directories:

```bash
MODELS_DIR=models_small_t30 T=30 OUT_ROOT=plots_small_t30 ./run_repr_analysis_multiseed.sh
```

## Additional analysis and reports

- `analysis_tables/` contains summarized experiment tables.
- `plot_*.py` scripts generate individual figures.
- `findings_report*.tex`, accompanying PDFs, and `SNN_Rotation_Invariance_Consolidated_Report.docx` document the experimental findings.
- `build_consolidated_report.py` builds the consolidated report.

## Repository contents

Generated checkpoints, plot images, local datasets, Python environments, logs, and LaTeX build artifacts are intentionally excluded from version control. This keeps the repository focused on source code, analysis tables, and written reports.
