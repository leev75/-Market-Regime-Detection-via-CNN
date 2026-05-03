# 📈 Market Regime Detection via CNN on Cross-Asset Correlation Heatmaps

> Detecting Bull, Bear, and Neutral market regimes by treating correlation structure as an image classification problem.

---
## poster pdf : 
![Poster](Market Regime CNN — A1 Landscape.pdf)



## Overview

Traditional regime detection relies on univariate thresholds — a single VIX level, a single drawdown percentage. This project takes a different approach: it captures the **structural relationship between asset classes** as a visual signal and trains a Convolutional Neural Network to classify the current market regime.

Every 30 trading days, a Pearson correlation matrix across 10 assets is rendered as a 64×64 heatmap image. The CNN learns to distinguish the tight, crisis-driven co-movement of a Bear market from the diversified, lower-correlation structure of a Bull market — purely from the visual pattern of the heatmap.

This project is one half of a **dual-modality regime detection system**:
- 🧠 **This repo** — structural/quantitative signals via CNN on correlation heatmaps
- 💬 **Companion repo** — qualitative/sentiment signals via FinBERT on financial news

---

## Method

### Assets
10 assets covering equity sectors, commodities, fixed income, volatility, and crypto:

| Ticker | Description |
|--------|-------------|
| XLF | Financials |
| XLK | Technology |
| XLE | Energy |
| XLV | Health Care |
| XLI | Industrials |
| GLD | Gold |
| TLT | 20+ Year Treasury Bonds |
| VNQ | Real Estate |
| ^VIX | Volatility Index |
| BTC-USD | Bitcoin |

### Pipeline

```
Daily Returns (yfinance)
        ↓
Rolling 30-day Pearson Correlation Matrix
        ↓
64×64 Heatmap Image (seaborn, vmin=-1, vmax=+1)
        ↓
CNN Classifier
        ↓
Regime Label: Bull / Bear / Neutral
```

### Labelling Logic

| Regime | Condition |
|--------|-----------|
| **Bull** | VIX < 20 AND S&P 500 drawdown < 5% |
| **Bear** | VIX > 30 OR S&P 500 drawdown > 15% |
| **Neutral** | Everything else |

### Architecture

```
Input: 64×64×3 RGB heatmap
  → Conv2D(32) + MaxPool
  → Conv2D(64) + MaxPool
  → Conv2D(128) + MaxPool
  → GlobalAveragePooling2D
  → Dense(64) + Dropout(0.4)
  → Softmax(3)
```

Also benchmarked against **MobileNetV2** as a transfer learning backbone.

### Train / Test Split

| Split | Period | Rationale |
|-------|--------|-----------|
| Train | 2005–2020 | Includes GFC, Euro crisis, COVID crash |
| Test | 2021–2024 | Fully unseen; post-training period |

Split is **strictly chronological** — no shuffling — to prevent lookahead bias.

---

## Key Results

| Metric | Value |
|--------|-------|
| Bear Recall (default threshold) | 9.8% |
| Bear Recall (t=0.10) | **61%** — no retraining required |
| Bear Average Precision | 0.542 vs 0.092 random → **5.9× lift** |
| Bull Average Precision | 0.809 |
| Custom CNN vs MobileNetV2 | **+14 F1 points** |

### Class Imbalance Handling
The training set is ~65% Bull. Class weights (Bear ×3.62, Neutral ×1.27, Bull ×0.52) are applied to the cross-entropy loss so rare-class errors penalise the network more heavily.

### Threshold Calibration
Rather than retraining, the Bear decision threshold is shifted from 0.50 → **0.10**, maximising Bear recall while keeping macro-F1 above the argmax baseline. This is the key operational insight: **threshold tuning is a free performance lever**.

---

## Visualisations

- **Sample heatmaps** per regime class — visual inspection of what the CNN sees
- **Confusion matrix** — breakdown of classification errors
- **Per-class F1 vs Bear threshold** — operating point selection
- **Precision-Recall curves** — lift over random baseline per class
- **Grad-CAM** — saliency maps showing which regions of the heatmap drive predictions
- **S&P 500 regime timeline overlay** — predicted regimes plotted against price history

---

## Project Structure

```
├── notebook.ipynb          # Full pipeline: data → heatmaps → training → evaluation
├── README.md
```

---

## Requirements

```
yfinance
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```

Install all with:

```bash
pip install yfinance tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
```

The notebook is designed to run end-to-end on **Google Colab** with no local setup required.

---

## Limitations

- Labelling thresholds (VIX 20/30, drawdown 5%/15%) are rule-based approximations — not ground truth
- The neutral class has a genuinely ambiguous visual signature; even with class weighting, it remains the hardest to separate
- Model has not been tested on live/streaming data
- BTC-USD data only available from ~2014; pre-2014 correlation matrices use 9 assets

---

## Context

This project was developed as part of a university machine learning coursework, with a research poster and codebase as deliverables. The brief required real-world business logic and prohibited standard benchmark datasets (MNIST, CIFAR, ImageNet).

---

## Author

Built by [Khelil Dhiaeddine] · [university of saad dahlab blida] · 2025-2026
[LinkedIn](www.linkedin.com/in/dhiaeddine-khelil-70a4163a9) · [Email](Dheakhelil2004@gmail.com)
