# MSc in AI Capstone #4 - Deep Learning Systems: Landmark Image Classification

**Repository:** https://github.com/PHIacademy/deep-learning-systems.git

A convolutional neural network (CNN), trained from scratch in PyTorch, that classifies photographs into one of 50 landmark categories. This project implements a baseline model, an experimental configuration that isolates a single change to the training setup, and a direct, evidence-based comparison of the two.

## Overview

- **Task type:** Image classification (CNN)
- **Dataset:** A curated 50-class, ~6,000-image subset of [Google's Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark), distributed for coursework use via Udacity's Deep Learning Nanodegree
- **Baseline model:** A 5-block CNN (Conv → BatchNorm → ReLU → MaxPool) with a dropout-regularized fully-connected head, trained with SGD
- **Experimental model:** Identical architecture and data pipeline, with the optimizer changed to Adam — the single variable isolated for comparison
- **Result:** Baseline (SGD) reached **41.9% test accuracy**; experimental (Adam) reached **21.3%**, while still improving at the training cutoff — see [Results](#results) below

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data.py            # Data loading, transforms, batch visualization
│   ├── helpers.py         # Dataset download/extraction, normalization stats, plotting
│   ├── model.py            # MyModel — the CNN architecture
│   ├── optimization.py     # Loss and optimizer factory functions
│   ├── predictor.py         # Inference wrapper (TorchScript export)
│   └── train.py             # Training, validation, and test loops
├── deep_learning.ipynb                             # Main notebook: data → baseline → experiment → evaluation
├── Deep_Learning_Systems_Analysis_Report.pdf        # Written analysis report (APA 7 format)
├── requirements.txt                                 # Reproducibility — exact package versions used
└── README.md
```

## Dataset

A 50-class, ~6,000-image subset of Google's Landmarks Dataset v2 (Weyand et al., 2020), pre-split into:
- **Training/validation:** 4,996 images
- **Test:** 1,250 images

Class labels are given by folder names combining a numeric landmark ID with a human-readable name (e.g. `07.Stonehenge`). The dataset is downloaded and extracted automatically the first time the notebook runs (see [Setup](#setup)).

## Setup

**Requirements:** Python 3.8+, and a CUDA-capable GPU (recommended — the notebook will also run on CPU, just considerably slower).

```bash
git clone https://github.com/PHIacademy/deep-learning-systems.git
cd deep-learning-systems
pip install -r requirements.txt
```

Or, from within the notebook itself:

```python
%pip install -r requirements.txt
```

## Usage

Open `deep_learning.ipynb` and run top to bottom. The notebook is organized into the following stages:

1. **Environment checks** — confirms required libraries import correctly and detects GPU/CPU
2. **Dataset setup** — downloads and extracts the dataset, computes normalization statistics
3. **Data inspection** — visualizes a batch, inspects shapes/labels/dtypes, notes data-quality considerations
4. **Baseline model** — defines and trains `MyModel` with SGD
5. **Experimental comparison** — retrains an identical model with Adam, documenting the single changed variable
6. **Evaluation** — quantitative comparison (loss, accuracy) and qualitative comparison (sample predictions) between both configurations
7. **Summary** — key findings and caveats

The cell of `src/helpers.py::setup_env()` handles dataset download and extraction automatically; no manual data preparation is required.

## Results

| Configuration | Optimizer | Final Train Loss | Final Valid Loss | Test Loss | Test Accuracy |
|---|---|---|---|---|---|
| Baseline | SGD (lr=0.01) | 2.486 | 2.368 | 2.200 | **41.9%** |
| Experimental | Adam (lr=0.001) | 3.395 | 3.217 | 3.109 | **21.3%** |

The baseline converged steadily within the 30-epoch training budget and had largely leveled off by the final epochs. The experimental (Adam) configuration required a lower learning rate than the baseline's to train stably at all — using the baseline's learning rate caused it to diverge entirely — and, once corrected, was **still improving** at the epoch cutoff with no sign of plateauing. The observed gap therefore reflects convergence speed under a fixed training budget at least as much as it reflects a ceiling difference between the two optimizers. Full discussion, including the divergence investigation, is in the accompanying analysis report.

## Key Findings

- Batch normalization, dropout before the final linear layer, and progressive channel-depth scaling (32→512) were the key design choices behind the baseline architecture.
- Optimizer choice is not a drop-in swap: Adam required its own, separately-tuned learning rate to train stably — reusing the SGD learning rate caused outright divergence.
- A single-run comparison carries real variance; a controlled re-seed was used to confirm the experimental model's behavior was reproducible rather than an artifact of initialization.

## Limitations

- Single training run per configuration (no multi-seed averaging)
- Modest per-class sample size (~100–140 images/class) limits what a from-scratch CNN can learn without transfer learning
- Evaluation relies on top-1 accuracy and loss only; no per-class or per-region breakdown

See the analysis report for a full discussion of limitations, ethical considerations, and future improvements.

## Reproducibility

`requirements.txt` was generated via `pip freeze` after a full successful run of `deep_learning.ipynb`, capturing the exact package versions used.

## References

- Weyand, T., Araujo, A., Cao, B., & Sim, J. (2020). Google Landmarks Dataset v2: A large-scale benchmark for instance-level recognition and retrieval. *CVPR 2020.*
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015.*
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR, 15*(1), 1929–1958.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015.*

Full citation list is available in `Deep_Learning_Systems_Analysis_Report.pdf`.

## License

This project was completed as part of an MSc in Artificial Intelligence capstone assignment. The dataset subset used is distributed for coursework/academic use; see the analysis report for full sourcing details.
