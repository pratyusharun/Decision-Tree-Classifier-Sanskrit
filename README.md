# 🕉️ Sanskrit Character Recognition using Decision Tree Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-73.8%25-brightgreen.svg)](#benchmark--evaluation-metrics)

A complete Machine Learning system for recognizing handwritten Sanskrit / Devanagari characters using a **Decision Tree Classifier**. The pipeline includes robust character bounding-box preprocessing, multi-faceted spatial feature extraction, model evaluation across 62 classes, a standalone **Google Colab Notebook**, an exported `.joblib` model bundle, and a **single-command CLI prediction tool** that accepts any image URL or local file path.

---

## 📌 Dataset Overview

The system is trained and evaluated on the **Sanskrit MNIST** dataset, consisting of **31,000 images** across **62 balanced character classes** (500 images per class):

- **15 Vowels (*Svara*) & Modifiers (*Yogavaha*)**: `अ`, `आ`, `इ`, `ई`, `उ`, `ऊ`, `ऋ`, `ॠ`, `ऌ`, `ए`, `ऐ`, `ओ`, `औ`, `अं`, `अः`
- **33 Consonants (*Vyanjana*)**: `क`, `ख`, `ग`, `घ`, `ङ`, `च`, `छ`, `ज`, `झ`, `ञ`, `ट`, `ठ`, `ड`, `ढ`, `ण`, `त`, `थ`, `द`, `ध`, `न`, `प`, `फ`, `ब`, `भ`, `म`, `य`, `र`, `ल`, `व`, `श`, `ष`, `स`, `ह`
- **4 Conjuncts & Vedic sounds**: `ळ`, `क्ष`, `ज्ञ`, `त्र`
- **10 Devanagari Numerals (*Anka*)**: `०`, `१`, `२`, `३`, `४`, `५`, `६`, `७`, `८`, `९`

---

## ⚡ Quickstart (Inference in 3 Steps)

Anyone viewing your GitHub repo can clone the repository, install dependencies, and classify any Sanskrit character from a web image URL in a single terminal command:

### 1. Clone the repository
```bash
git clone https://github.com/pratyusharun/Decision-Tree-Classifier-Sanskrit.git
cd Decision-Tree-Classifier-Sanskrit
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run prediction with an image URL or local path
```bash
# Classify an online image URL:
python predict.py --image "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png"

# Or classify a local image file:
python predict.py images/15_ka/15_0000.png
```

#### Example CLI Output:
```text
=================================================================
       SANSKRIT CHARACTER CLASSIFIER - PREDICTION
=================================================================
[*] Input Source: images/15_ka/15_0000.png
[*] Model Loaded: sanskrit_dt_model.joblib

[*] Preprocessed Character Canvas (28x28 centered):
--------------------------------
                            
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░     
   ░░▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒░░    
     ░░░▒▒▒▒▒▓▓▒░░          
     ░▒▒▒░░▒▒▓▓▓▒▒▒░░       
     ▒▓▒░░░▒▒▓▓▓▒▒▒▒░░      
     ░▒▒░░▒▒▓▓▓░ ░░▒▒▒░     
       ░▒▒▒▓▓▓▓░ ░▒▒▒░      
            ░▒▒░            
                            
--------------------------------

=================================================================
                      PREDICTION RESULT
=================================================================
   Devanagari Character:   क
   Transliteration (IAST): ka
   Description:            Velar Consonant ka
   Category:               Consonant (Vyanjana)
   Class Label:            15_ka
   Confidence:             100.00%
=================================================================

[*] Top 3 Candidates:
    Rank  | Class      | Char  | Translit   | Confidence
    --------------------------------------------------
    1     | 15_ka      | क     | ka         |   100.00%
    2     | 61_9       | ९     | 9 (nava)   |     0.00%
    3     | 59_7       | ७     | 7 (sapta)  |     0.00%
=================================================================
```

---

## 🧠 Engineering & Architecture

### Why Feature Engineering Matters for Decision Trees:
A standard Decision Tree relies on axis-aligned orthogonal splits (e.g. `pixel_240 <= 0.4`).
- **Naive Raw Pixels**: When trained on raw flattened pixels, Decision Trees achieve only **~10-15% test accuracy** across 62 classes due to translation and stroke shifts.
- **Our Multi-Faceted Pipeline**:
  1. **Stroke Polarity Detection**: Auto-detects whether the background is light or dark and normalizes strokes to positive foreground values.
  2. **Bounding-Box Isolation**: Finds the active stroke boundary and applies an aspect-preserving resize with centered padding into a 28x28 canvas.
  3. **856-Dimensional Feature Vector**:
     - **784 Normalized Pixel Values** ($28 \times 28$)
     - **16 Spatial Zone Densities** ($4 \times 4$ grid quadrant averages)
     - **28 Horizontal Row Projections** (stroke distribution along rows)
     - **28 Vertical Column Projections** (stroke distribution along columns)

This preprocessing pipeline boosts Decision Tree test accuracy from **~10% to 73.79%** on 6,200 unseen test images!

---

## 📊 Benchmark & Evaluation Metrics

Evaluated on a **stratified 20% unseen test split** (6,200 character images) across all 62 classes:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **73.79%** |
| **Precision (Macro)** | **74.25%** |
| **Recall (Macro)** | **73.79%** |
| **F1-Score (Macro)** | **73.79%** |
| **Precision (Weighted)** | **74.25%** |
| **Recall (Weighted)** | **73.79%** |
| **F1-Score (Weighted)** | **73.79%** |
| **Feature Extraction Time** | ~15.5 seconds (31,000 images) |
| **Model Training Time** | ~16.5 seconds |
| **Model Export Size** | 2.99 MB (`sanskrit_dt_model.joblib`) |

### Top Confused Character Pairs (Linguistic & Visual Similarity):
The characters with occasional confusion exhibit extreme visual and orthographic similarity in Devanagari handwriting:
1. **ॠ (`07_rii`) vs ऋ (`06_ri`)**: Differ only by an extra bottom vowel stroke.
2. **य (`40_ya`) vs प (`35_pa`)**: Differ only by the left loop curvature.
3. **ड (`27_dda`) vs ङ (`19_nga`)**: Differ only by a small diacritical dot.
4. **ह (`47_ha`) vs इ (`02_i`)**: Share identical upper stroke and spine.
5. **ध (`33_dha`) vs थ (`31_tha`)**: Differ only by the upper shirorekha break and loop.

The full 62x62 Confusion Matrix is saved in `confusion_matrix.png`.

---

## 📁 Repository Structure

```text
├── images/                           # Sanskrit MNIST dataset (62 classes, 500 images each)
│   ├── 00_a/
│   ├── 01_aa/
│   └── ...
├── sanskrit_mnist_decision_tree.ipynb # Full Google Colab Jupyter Notebook
├── sanskrit_utils.py                  # Character metadata, preprocessing & feature extraction
├── train.py                           # Training & evaluation script (exports model & metrics)
├── predict.py                         # Single-line CLI inference for URLs and local images
├── sanskrit_dt_model.joblib           # Pretrained Decision Tree model bundle (2.99 MB)
├── confusion_matrix.png               # High-resolution 62x62 confusion matrix heatmap
├── classification_report.txt          # Detailed per-class precision/recall/F1 metrics
├── requirements.txt                   # Python package dependencies
└── README.md                          # Project documentation
```

---

## 🏋️ Training the Model Locally

To retrain the model on the dataset:
```bash
python train.py
```

Optional arguments:
- `--samples_per_class 200`: Train on a subset of samples per class for ultra-fast experimentation.
- `--model_output custom_model.joblib`: Specify a custom output model path.
- `--cm_output custom_cm.png`: Specify output location for the confusion matrix heatmap.

---

## 🌐 Running in Google Colab

1. Open `sanskrit_mnist_decision_tree.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially. The notebook walks through EDA, visualization, feature engineering, Decision Tree training, evaluation, model export, and interactive URL prediction with matplotlib bar charts.

---

## 📜 License
This project is open-source under the MIT License.
