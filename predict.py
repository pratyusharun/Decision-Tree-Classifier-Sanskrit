"""
Sanskrit Character Prediction CLI
Loads trained Decision Tree model bundle and predicts the Sanskrit character
from a local image file or an online image URL in a single command line.

Usage:
    python predict.py --image "https://example.com/sanskrit_char.png"
    python predict.py "https://example.com/sanskrit_char.png"
    python predict.py images/15_ka/15_0000.png
"""

import sys
import os
import argparse
import joblib
import numpy as np

# Ensure Windows console handles UTF-8 (Devanagari characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sanskrit_utils import SANSKRIT_MAPPING, preprocess_image, extract_features, render_ascii_preview

def predict_character(image_source, model_path="sanskrit_dt_model.joblib", top_k=3, show_ascii=True):
    """
    Loads model, extracts features, and predicts the Sanskrit character.
    """
    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' not found! Please train the model first by running `python train.py`.")
        sys.exit(1)
        
    # Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = bundle["classes"]
    mapping = bundle.get("mapping", SANSKRIT_MAPPING)
    
    print("=" * 65)
    print("       SANSKRIT CHARACTER CLASSIFIER - PREDICTION")
    print("=" * 65)
    print(f"[*] Input Source: {image_source}")
    print(f"[*] Model Loaded: {model_path}")
    
    # Preprocess & Extract Features
    try:
        processed_arr = preprocess_image(image_source)
        features = extract_features(image_source)
    except Exception as e:
        print(f"\n[Error] Failed to load/preprocess image from '{image_source}': {e}")
        sys.exit(1)
        
    # ASCII visual preview
    if show_ascii:
        print("\n[*] Preprocessed Character Canvas (28x28 centered):")
        print("-" * 32)
        print(render_ascii_preview(processed_arr))
        print("-" * 32)
        
    # Model inference
    feat_2d = features.reshape(1, -1)
    probabilities = model.predict_proba(feat_2d)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    best_idx = top_indices[0]
    best_class = classes[best_idx]
    best_prob = probabilities[best_idx] * 100.0
    
    best_meta = mapping.get(best_class, {})
    devanagari_char = best_meta.get("devanagari", "?")
    iast_name = best_meta.get("iast", best_class)
    char_desc = best_meta.get("name", best_class)
    category = best_meta.get("category", "Unknown")
    
    print("\n" + "=" * 65)
    print("                      PREDICTION RESULT")
    print("=" * 65)
    print(f"   Devanagari Character:   {devanagari_char}")
    print(f"   Transliteration (IAST): {iast_name}")
    print(f"   Description:            {char_desc}")
    print(f"   Category:               {category}")
    print(f"   Class Label:            {best_class}")
    print(f"   Confidence:             {best_prob:.2f}%")
    print("=" * 65)
    
    print(f"\n[*] Top {top_k} Candidates:")
    print(f"    {'Rank':<5} | {'Class':<10} | {'Char':<5} | {'Translit':<10} | {'Confidence':<10}")
    print("    " + "-" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        c_name = classes[idx]
        c_meta = mapping.get(c_name, {})
        c_char = c_meta.get("devanagari", "?")
        c_iast = c_meta.get("iast", c_name)
        c_prob = probabilities[idx] * 100.0
        print(f"    {rank:<5} | {c_name:<10} | {c_char:<5} | {c_iast:<10} | {c_prob:>8.2f}%")
    print("=" * 65 + "\n")
    
    return {
        "class": best_class,
        "devanagari": devanagari_char,
        "iast": iast_name,
        "confidence": best_prob,
        "top_candidates": [
            {
                "rank": r,
                "class": classes[i],
                "char": mapping.get(classes[i], {}).get("devanagari", "?"),
                "prob": float(probabilities[i])
            }
            for r, i in enumerate(top_indices, 1)
        ]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Classify a Sanskrit character image from a URL or local file path.",
        epilog="Example: python predict.py --image https://example.com/sanskrit.png"
    )
    # Support both --image and positional argument
    parser.add_argument("positional_image", nargs="?", default=None, help="Path or URL to input image (optional if --image is used)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path or URL to input image")
    parser.add_argument("--model", "-m", type=str, default="sanskrit_dt_model.joblib", help="Path to trained model bundle")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of top candidates to display (default: 3)")
    parser.add_argument("--no_ascii", action="store_true", help="Disable terminal ASCII preview")
    
    args = parser.parse_args()
    
    image_src = args.image or args.positional_image
    if not image_src:
        parser.print_help()
        print("\n[Error] Please provide an image URL or local file path via `--image <URL_or_path>` or positional argument.")
        sys.exit(1)
        
    predict_character(
        image_source=image_src,
        model_path=args.model,
        top_k=args.top_k,
        show_ascii=not args.no_ascii
    )

if __name__ == "__main__":
    main()
