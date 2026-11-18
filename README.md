# Learning to Verify

Code for the signature-guided dual-model verifier and its detector-aware evaluation. The repository contains **source code only**. It does not include datasets, experiment results, trained models, or manuscript files.

## Requirements

- Python 3.11 or newer; an NVIDIA GPU is strongly recommended for the full study.
- Install a matching PyTorch and torchvision pair for your machine using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the remaining packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The code downloads CIFAR-10, SVHN, and STL-10 through torchvision when first run. Downloads go to `data/`; trained classifiers go to `checkpoints/`; all experiment artifacts go to `Results/`. These paths are ignored by Git.

## Run or resume the paper experiments

From the repository root, run:

```powershell
python run_paper_experiments.py --device auto
```

Run the **same command again** after an interruption. Completed artifacts are saved atomically and have `.complete` markers; the runner skips completed stages. Verifier training resumes from its last saved epoch. `--device auto` selects a CUDA GPU when available, otherwise CPU. The complete study is computationally expensive and is not intended as a quick test.

The one-command runner performs, in order:

1. The ResNet-18 primary matrix for CIFAR-10, SVHN, and STL-10 with seeds 0, 1, and 2, four direct adaptive attacks, directly attacked score baselines, and the original component ablations.
2. Full versus no-signature verifier evaluations on CIFAR-10 and SVHN, three seeds and four attacks per dataset.
3. Direct cross-norm $L_2$ PGD on CIFAR-10 and SVHN, seed 0, for both verifier variants.
4. DenseNet-121 on CIFAR-10, seed 0: full verifier with direct $L_\infty$ and $L_2$ PGD; a 200-example no-signature $L_\infty$ control; and the matched 200-example summary.

The DenseNet-121 control uses 20 successful targeted examples per class. Compare it with the **matched 200-example subset**, not the separate 1,000-example full-method evaluation. The main runner's `minimum` manifest describes its original two-seed protocol; this wrapper explicitly runs seed 2 as an additional primary run, and each run's artifacts are preserved separately.

To run one phase, use `--stage primary`, `controls`, `crossnorm`, or `architecture`. Later phases depend on classifiers or verifiers produced by earlier phases. For a smaller first check:

```powershell
python run_publication.py --smoke --train-missing-classifiers --skip-ablations --skip-cross-architecture-transfer
```

The smoke run writes to separate paths and is **not** a scientific result. For one full primary configuration:

```powershell
python run_publication.py --datasets cifar10 --architectures resnet18 --seeds 0 --train-missing-classifiers --device auto
```

## Evaluation notes

The main method keeps the classifier fixed while training a second verifier with short targeted PGD and class-specific activation guidance. At inference it scores `1 - p_verifier(classifier_prediction)` using a class-calibrated threshold; the signature bank is not needed. The no-signature control sets the geometry weight to zero while retaining the same verifier training and inference rule.

Evaluation attacks optimize against the detector they are evaluating. The AutoAttack entry means **standard AutoAttack candidates followed by targeted detector-aware PGD refinement**; it is not official detector-aware AutoAttack. Reports record attack success, perturbation checks, and bypass conditional on successful targeted attacks. Do not interpret unfinished runs or invalid attack checks as completed results.

The paper's released numbers should be interpreted using their reported dataset, architecture, seed, norm, and sample count. Rerunning from scratch can change numeric outcomes across hardware and package versions.
