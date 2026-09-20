"""
Sanskrit Character Recognition - Decision Tree Training & Evaluation Script
Trains a Decision Tree classifier on Sanskrit MNIST characters, evaluates metrics,
generates a confusion matrix heatmap, and exports the model bundle using joblib.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np

# Ensure Windows terminal handles UTF-8 (Devanagari characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sanskrit_utils import SANSKRIT_MAPPING, preprocess_image, extract_features

def load_dataset(data_dir="images", samples_per_class=None):
    """
    Loads images from the class directories and extracts engineered features.
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"[*] Found {len(classes)} classes in '{data_dir}'")
    
    X_list = []
    y_list = []
    
    total_imgs = 0
    t0 = time.time()
    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        filenames = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if samples_per_class:
            filenames = filenames[:samples_per_class]
            
        for fname in filenames:
            fpath = os.path.join(class_folder, fname)
            feat = extract_features(fpath)
            X_list.append(feat)
            y_list.append(class_idx)
            total_imgs += 1
            
        if (class_idx + 1) % 10 == 0 or class_idx == len(classes) - 1:
            print(f"    Loaded {class_idx + 1}/{len(classes)} classes ({total_imgs} images processed)...")
            
    print(f"[+] Feature extraction completed in {time.time() - t0:.2f}s.")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, classes

def train_and_evaluate(
    data_dir="images",
    samples_per_class=None,
    model_output="sanskrit_dt_model.joblib",
    cm_output="confusion_matrix.png",
    random_state=42
):
    print("=" * 70)
    print("  SANSKRIT CHARACTER RECOGNITION - DECISION TREE CLASSIFIER")
    print("=" * 70)
    
    # 1. Load Data & Prepare Features
    X, y, classes = load_dataset(data_dir=data_dir, samples_per_class=samples_per_class)
    print(f"[*] Total dataset shape: X = {X.shape}, y = {y.shape}")
    
    # 2. Stratified Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"[*] Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # 3. Train Decision Tree Classifier
    print("[*] Training Decision Tree Classifier...")
    dt_clf = DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
    )
    t_train = time.time()
    dt_clf.fit(X_train, y_train)
    train_duration = time.time() - t_train
    print(f"[+] Model trained in {train_duration:.2f}s.")
    
    # 4. Evaluation on Training & Test Sets
    y_train_pred = dt_clf.predict(X_train)
    y_test_pred = dt_clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    
    precision_weighted = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    
    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE METRICS (ON UNSEEN TEST DATA)")
    print("=" * 70)
    print(f"  Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"  Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"  Precision (Macro):     {precision_macro * 100:.2f}%")
    print(f"  Recall (Macro):        {recall_macro * 100:.2f}%")
    print(f"  F1-Score (Macro):      {f1_macro * 100:.2f}%")
    print(f"  Precision (Weighted):  {precision_weighted * 100:.2f}%")
    print(f"  Recall (Weighted):     {recall_weighted * 100:.2f}%")
    print(f"  F1-Score (Weighted):   {f1_weighted * 100:.2f}%")
    print("=" * 70)
    
    # Per-class summary table preview
    labels_names = [classes[i] for i in range(len(classes))]
    rep = classification_report(y_test, y_test_pred, target_names=labels_names, digits=4, zero_division=0)
    
    report_file = "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Sanskrit Character Recognition - Decision Tree Classification Report\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}\n\n")
        f.write(rep)
    print(f"[+] Full classification report saved to '{report_file}'.")
    
    # 5. Confusion Matrix Analysis & Heatmap Plot
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Identify Top Confused Character Pairs
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    confused_indices = np.unravel_index(np.argsort(cm_off_diag, axis=None)[::-1], cm_off_diag.shape)
    
    print("\n[*] Top 5 Confused Character Pairs:")
    for rank in range(min(5, len(confused_indices[0]))):
        i, j = confused_indices[0][rank], confused_indices[1][rank]
        count = cm_off_diag[i, j]
        if count > 0:
            char_i = SANSKRIT_MAPPING.get(classes[i], {}).get("devanagari", classes[i])
            char_j = SANSKRIT_MAPPING.get(classes[j], {}).get("devanagari", classes[j])
            print(f"    {rank+1}. True: '{classes[i]}' ({char_i}) -> Pred: '{classes[j]}' ({char_j}) | Count: {count}")
            
    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(18, 16))
    sns.set_theme(style="white")
    display_labels = [f"{classes[i]} ({SANSKRIT_MAPPING.get(classes[i], {}).get('iast', '')})" for i in range(len(classes))]
    
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Sample Count"}
    )
    plt.title(f"Sanskrit MNIST Decision Tree - Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Character Class", fontsize=12)
    plt.ylabel("True Character Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(cm_output, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix heatmap saved to '{cm_output}'.")
    
    # 6. Export Model Bundle with Joblib
    model_bundle = {
        "model": dt_clf,
        "classes": classes,
        "mapping": SANSKRIT_MAPPING,
        "feature_length": X.shape[1],
        "target_size": (28, 28),
        "metrics": {
            "test_accuracy": float(test_acc),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted)
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_bundle, model_output)
    model_size_mb = os.path.getsize(model_output) / (1024 * 1024)
    print(f"[+] Successfully exported trained model bundle to '{model_output}' ({model_size_mb:.2f} MB).")
    print("=" * 70)
    return model_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sanskrit Character Decision Tree Classifier")
    parser.add_argument("--data_dir", type=str, default="images", help="Path to images directory")
    parser.add_argument("--samples_per_class", type=int, default=None, help="Number of samples per class (default: all 500)")
    parser.add_argument("--model_output", type=str, default="sanskrit_dt_model.joblib", help="Output path for joblib model bundle")
    parser.add_argument("--cm_output", type=str, default="confusion_matrix.png", help="Output path for confusion matrix heatmap")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed")
    
    args = parser.parse_args()
    train_and_evaluate(
        data_dir=args.data_dir,
        samples_per_class=args.samples_per_class,
        model_output=args.model_output,
        cm_output=args.cm_output,
        random_state=args.random_state
    )
