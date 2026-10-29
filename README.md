# ML_-PROJECT
# EEG-Based Emotion Recognition with MSPCA, MCOV and Ensemble Classifiers

A re-implementation of **Subasi & Qaisar (2024), "EEG-based emotion recognition using modified covariance and ensemble classifiers"** (*Journal of Ambient Intelligence and Humanized Computing*, 15:575–591, [doi:10.1007/s12652-023-04715-5](https://doi.org/10.1007/s12652-023-04715-5)).

**Author:** Anuj Kumar Sarki

## Pipeline

```
EEG segment -> MSPCA denoising -> MCOV features -> Ensemble classifier -> Emotion
```

1. **MSPCA denoising.** Symlets-4 wavelet, 5 levels, PCA across channels at each scale, then wavelet reconstruction.
2. **MCOV features.** Modified covariance autoregressive model (order 15) per channel, converted to a power spectral density at 40 frequencies between 0 and 75 Hz.
3. **Classifiers.** Decision Tree, Random Forest, SVM and Bagging + Random Forest.

Each instance is a 1,024-sample (5.12 s) segment with 62 channels. The classifiers receive log10 of the PSD: 62 x 40 = **2,480 features** per instance.

## Dataset

The paper uses SEED (3 emotions, 15 subjects). SEED needs an application to the BCMI lab, so this project uses a Kaggle copy of **SEED-IV** (`phhasian0710/seed-iv`). Its provenance and licence have not been verified.

| | Paper | This project |
|---|---|---|
| Dataset | SEED | SEED-IV subset |
| Emotion classes | 3 (negative, neutral, positive) | 4 (neutral, sad, fear, happy) |
| Subjects | 15 | 5 (subjects 1–5) |
| Sessions | 3 | 1 |
| Instances | not stated | 1,410 segments |

## Results

Accuracy (chance = 0.25):

| Model | Paper (SEED, 10-fold) | Random 10-fold | Trial-grouped 5-fold | Leave-one-subject-out |
|---|---|---|---|---|
| Decision Tree | 0.809 | 0.460 | 0.343 | 0.270 |
| Random Forest | 0.878 | 0.667 | 0.433 | 0.309 |
| SVM | 0.889 | 0.728 | 0.506 | 0.343 |
| Bagging + RF | 0.996 | 0.641 | 0.425 | 0.323 |


Key observations:
- Random 10-fold splits put segments from the same trial in both training and test sets, so they overestimate real performance.
- On unseen subjects (leave-one-subject-out) every model is only a few points above chance, and the confidence intervals of SVM, Random Forest and Bagging + RF overlap.
- A Random Forest can identify *which subject* a segment came from with 92.6% accuracy (chance 0.20). The features carry a strong subject signature, which explains the gap between random and cross-subject results.

## Differences from the paper

- SVM uses an RBF kernel (C = 100) in scikit-learn; the paper used a Puk kernel in Weka.
- Decision Tree is CART, not C4.5.
- Bagging + RF uses 10 bags of 10-tree forests (reduced from 100 trees for Colab time).
- The paper does not define an "instance" or the PSD frequency grid, so both are my choices.
- No artifact rejection was applied (some recordings contain very large amplitudes).

## Run it

1. Open `ml_made.ipynb` in Google Colab.
2. Upload your own `kaggle.json` when prompted (never commit it to GitHub).
3. Run the cells in order. The notebook downloads the data, denoises, extracts features, runs the three evaluation schemes, and saves results to `results/`, models to `models/`, and a zip bundle of the outputs.

Requirements: Python 3, numpy, scipy, pandas, scikit-learn, PyWavelets, matplotlib, joblib, kaggle.

## Reference

Subasi, A., & Qaisar, S. M. (2024). EEG-based emotion recognition using modified covariance and ensemble classifiers. *Journal of Ambient Intelligence and Humanized Computing*, 15, 575–591.
