"""
Sanskrit Character Recognition Utilities
Mapping, image preprocessing, feature extraction, and visualization.
"""

import io
import os
import requests
import numpy as np
from PIL import Image

# 62-Class Sanskrit Character Metadata Mapping
SANSKRIT_MAPPING = {
    "00_a": {"devanagari": "अ", "iast": "a", "name": "Vowel a", "category": "Vowel (Svara)"},
    "01_aa": {"devanagari": "आ", "iast": "ā", "name": "Vowel ā", "category": "Vowel (Svara)"},
    "02_i": {"devanagari": "इ", "iast": "i", "name": "Vowel i", "category": "Vowel (Svara)"},
    "03_ii": {"devanagari": "ई", "iast": "ī", "name": "Vowel ī", "category": "Vowel (Svara)"},
    "04_u": {"devanagari": "उ", "iast": "u", "name": "Vowel u", "category": "Vowel (Svara)"},
    "05_uu": {"devanagari": "ऊ", "iast": "ū", "name": "Vowel ū", "category": "Vowel (Svara)"},
    "06_ri": {"devanagari": "ऋ", "iast": "ṛ", "name": "Vowel ṛ", "category": "Vowel (Svara)"},
    "07_rii": {"devanagari": "ॠ", "iast": "ṝ", "name": "Vowel ṝ", "category": "Vowel (Svara)"},
    "08_li": {"devanagari": "ऌ", "iast": "ḷ", "name": "Vowel ḷ", "category": "Vowel (Svara)"},
    "09_e": {"devanagari": "ए", "iast": "e", "name": "Vowel e", "category": "Vowel (Svara)"},
    "10_ai": {"devanagari": "ऐ", "iast": "ai", "name": "Vowel ai", "category": "Vowel (Svara)"},
    "11_o": {"devanagari": "ओ", "iast": "o", "name": "Vowel o", "category": "Vowel (Svara)"},
    "12_au": {"devanagari": "औ", "iast": "au", "name": "Vowel au", "category": "Vowel (Svara)"},
    "13_am": {"devanagari": "अं", "iast": "aṃ", "name": "Anusvara aṃ", "category": "Yogavaha (Modifier)"},
    "14_ah": {"devanagari": "अः", "iast": "aḥ", "name": "Visarga aḥ", "category": "Yogavaha (Modifier)"},
    "15_ka": {"devanagari": "क", "iast": "ka", "name": "Velar Consonant ka", "category": "Consonant (Vyanjana)"},
    "16_kha": {"devanagari": "ख", "iast": "kha", "name": "Velar Consonant kha", "category": "Consonant (Vyanjana)"},
    "17_ga": {"devanagari": "ग", "iast": "ga", "name": "Velar Consonant ga", "category": "Consonant (Vyanjana)"},
    "18_gha": {"devanagari": "घ", "iast": "gha", "name": "Velar Consonant gha", "category": "Consonant (Vyanjana)"},
    "19_nga": {"devanagari": "ङ", "iast": "ṅa", "name": "Velar Consonant ṅa", "category": "Consonant (Vyanjana)"},
    "20_ca": {"devanagari": "च", "iast": "ca", "name": "Palatal Consonant ca", "category": "Consonant (Vyanjana)"},
    "21_cha": {"devanagari": "छ", "iast": "cha", "name": "Palatal Consonant cha", "category": "Consonant (Vyanjana)"},
    "22_ja": {"devanagari": "ज", "iast": "ja", "name": "Palatal Consonant ja", "category": "Consonant (Vyanjana)"},
    "23_jha": {"devanagari": "झ", "iast": "jha", "name": "Palatal Consonant jha", "category": "Consonant (Vyanjana)"},
    "24_nya": {"devanagari": "ञ", "iast": "ña", "name": "Palatal Consonant ña", "category": "Consonant (Vyanjana)"},
    "25_tta": {"devanagari": "ट", "iast": "ṭa", "name": "Retroflex Consonant ṭa", "category": "Consonant (Vyanjana)"},
    "26_ttha": {"devanagari": "ठ", "iast": "ṭha", "name": "Retroflex Consonant ṭha", "category": "Consonant (Vyanjana)"},
    "27_dda": {"devanagari": "ड", "iast": "ḍa", "name": "Retroflex Consonant ḍa", "category": "Consonant (Vyanjana)"},
    "28_ddha": {"devanagari": "ढ", "iast": "ḍha", "name": "Retroflex Consonant ḍha", "category": "Consonant (Vyanjana)"},
    "29_nna": {"devanagari": "ण", "iast": "ṇa", "name": "Retroflex Consonant ṇa", "category": "Consonant (Vyanjana)"},
    "30_ta": {"devanagari": "त", "iast": "ta", "name": "Dental Consonant ta", "category": "Consonant (Vyanjana)"},
    "31_tha": {"devanagari": "थ", "iast": "tha", "name": "Dental Consonant tha", "category": "Consonant (Vyanjana)"},
    "32_da": {"devanagari": "द", "iast": "da", "name": "Dental Consonant da", "category": "Consonant (Vyanjana)"},
    "33_dha": {"devanagari": "ध", "iast": "dha", "name": "Dental Consonant dha", "category": "Consonant (Vyanjana)"},
    "34_na": {"devanagari": "न", "iast": "na", "name": "Dental Consonant na", "category": "Consonant (Vyanjana)"},
    "35_pa": {"devanagari": "प", "iast": "pa", "name": "Labial Consonant pa", "category": "Consonant (Vyanjana)"},
    "36_pha": {"devanagari": "फ", "iast": "pha", "name": "Labial Consonant pha", "category": "Consonant (Vyanjana)"},
    "37_ba": {"devanagari": "ब", "iast": "ba", "name": "Labial Consonant ba", "category": "Consonant (Vyanjana)"},
    "38_bha": {"devanagari": "भ", "iast": "bha", "name": "Labial Consonant bha", "category": "Consonant (Vyanjana)"},
    "39_ma": {"devanagari": "म", "iast": "ma", "name": "Labial Consonant ma", "category": "Consonant (Vyanjana)"},
    "40_ya": {"devanagari": "य", "iast": "ya", "name": "Semivowel ya", "category": "Consonant (Vyanjana)"},
    "41_ra": {"devanagari": "र", "iast": "ra", "name": "Semivowel ra", "category": "Consonant (Vyanjana)"},
    "42_la": {"devanagari": "ल", "iast": "la", "name": "Semivowel la", "category": "Consonant (Vyanjana)"},
    "43_va": {"devanagari": "व", "iast": "va", "name": "Semivowel va", "category": "Consonant (Vyanjana)"},
    "44_sha": {"devanagari": "श", "iast": "śa", "name": "Sibilant śa", "category": "Consonant (Vyanjana)"},
    "45_ssa": {"devanagari": "ष", "iast": "ṣa", "name": "Sibilant ṣa", "category": "Consonant (Vyanjana)"},
    "46_sa": {"devanagari": "स", "iast": "sa", "name": "Sibilant sa", "category": "Consonant (Vyanjana)"},
    "47_ha": {"devanagari": "ह", "iast": "ha", "name": "Aspirate ha", "category": "Consonant (Vyanjana)"},
    "48_lla": {"devanagari": "ळ", "iast": "ḷa", "name": "Retroflex ḷa", "category": "Consonant (Vyanjana)"},
    "49_ksha": {"devanagari": "क्ष", "iast": "kṣa", "name": "Conjunct kṣa", "category": "Conjunct (Samyukta)"},
    "50_jna": {"devanagari": "ज्ञ", "iast": "jña", "name": "Conjunct jña", "category": "Conjunct (Samyukta)"},
    "51_tra": {"devanagari": "त्र", "iast": "tra", "name": "Conjunct tra", "category": "Conjunct (Samyukta)"},
    "52_0": {"devanagari": "०", "iast": "0 (śūnya)", "name": "Numeral 0", "category": "Numeral (Anka)"},
    "53_1": {"devanagari": "१", "iast": "1 (eka)", "name": "Numeral 1", "category": "Numeral (Anka)"},
    "54_2": {"devanagari": "२", "iast": "2 (dvi)", "name": "Numeral 2", "category": "Numeral (Anka)"},
    "55_3": {"devanagari": "३", "iast": "3 (tri)", "name": "Numeral 3", "category": "Numeral (Anka)"},
    "56_4": {"devanagari": "४", "iast": "4 (catur)", "name": "Numeral 4", "category": "Numeral (Anka)"},
    "57_5": {"devanagari": "५", "iast": "5 (pañca)", "name": "Numeral 5", "category": "Numeral (Anka)"},
    "58_6": {"devanagari": "६", "iast": "6 (ṣaṣ)", "name": "Numeral 6", "category": "Numeral (Anka)"},
    "59_7": {"devanagari": "७", "iast": "7 (sapta)", "name": "Numeral 7", "category": "Numeral (Anka)"},
    "60_8": {"devanagari": "८", "iast": "8 (aṣṭa)", "name": "Numeral 8", "category": "Numeral (Anka)"},
    "61_9": {"devanagari": "९", "iast": "9 (nava)", "name": "Numeral 9", "category": "Numeral (Anka)"}
}

def load_image(image_input):
    """
    Loads an image from a URL, local file path, bytes, or PIL Image.
    Returns a grayscale PIL Image.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("L")
    
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            return Image.fromarray(image_input).convert("L")
        return Image.fromarray(image_input.astype(np.uint8)).convert("L")
    
    if isinstance(image_input, str):
        if image_input.startswith("http://") or image_input.startswith("https://"):
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(image_input, headers=headers, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("L")
        elif os.path.isfile(image_input):
            return Image.open(image_input).convert("L")
        else:
            raise FileNotFoundError(f"Input image path or URL not found: {image_input}")
            
    raise ValueError(f"Unsupported image input type: {type(image_input)}")

def preprocess_image(image_input, target_size=(28, 28), stroke_margin=3):
    """
    Standardized Sanskrit OCR character preprocessing:
    1. Grayscale conversion.
    2. Stroke polarity detection (ensuring character stroke is foreground > 0, background = 0).
    3. Character bounding-box detection & crop.
    4. Aspect-ratio-preserving resize into target canvas with margin padding.
    
    Returns:
        processed_arr (np.ndarray): 2D float32 array normalized to [0.0, 1.0] of shape target_size.
    """
    pil_img = load_image(image_input)
    arr = np.array(pil_img, dtype=np.float32)
    
    # Auto-detect stroke polarity:
    # Most documents/datasets have light background (~255) and dark strokes (~0).
    # If the corners/border are predominantly bright, invert so stroke is high value.
    corners = [arr[:3, :3], arr[:3, -3:], arr[-3:, :3], arr[-3:, -3:]]
    corner_mean = np.mean(corners)
    if corner_mean > 127:
        arr = 255.0 - arr
        
    # Threshold to find character stroke pixels
    binary = arr > 30.0
    if not np.any(binary):
        # Empty image fallback
        return np.zeros(target_size, dtype=np.float32)
        
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    crop = arr[rmin:rmax + 1, cmin:cmax + 1]
    h, w = crop.shape
    
    # Max target dimension leaving margin
    max_target = max(1, target_size[0] - 2 * stroke_margin)
    max_d = max(h, w)
    scale = max_target / max_d
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    crop_pil = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    resized = crop_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Pad into centered target_size canvas
    pad_img = Image.new("L", target_size, 0)
    pad_x = (target_size[1] - new_w) // 2
    pad_y = (target_size[0] - new_h) // 2
    pad_img.paste(resized, (pad_x, pad_y))
    
    processed_arr = np.array(pad_img, dtype=np.float32) / 255.0
    return processed_arr

def extract_features(image_input, target_size=(28, 28)):
    """
    Extracts a rich multi-representation feature vector for the Decision Tree:
    1. Normalized centered pixel intensities (28 x 28 = 784 features).
    2. Spatial 4x4 Zone Densities (16 features).
    3. Horizontal Row Projection Profiles (28 features).
    4. Vertical Column Projection Profiles (28 features).
    Total: 856 numerical features.
    
    Returns:
        features (np.ndarray): 1D float32 array of length 856.
    """
    processed_arr = preprocess_image(image_input, target_size=target_size)
    
    # 1. Flattened pixel vector (784 features)
    pixels = processed_arr.flatten()
    
    # 2. 4x4 Zone Densities (16 features)
    zh, zw = target_size[0] // 4, target_size[1] // 4
    zones = []
    for r in range(4):
        for c in range(4):
            zone = processed_arr[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw]
            zones.append(float(zone.mean()))
            
    # 3. Horizontal Row Projection Profile (28 features)
    row_proj = processed_arr.sum(axis=1) / float(target_size[1])
    
    # 4. Vertical Column Projection Profile (28 features)
    col_proj = processed_arr.sum(axis=0) / float(target_size[0])
    
    features = np.concatenate([pixels, zones, row_proj, col_proj]).astype(np.float32)
    return features

def render_ascii_preview(processed_arr, width=28):
    """
    Renders a small ASCII character art for CLI visualization.
    """
    chars = [" ", "░", "▒", "▓", "█"]
    lines = []
    # Subsample or render directly
    h, w = processed_arr.shape
    for y in range(0, h, 2):  # step by 2 vertically to compensate for terminal aspect ratio
        line = ""
        for x in range(w):
            val = processed_arr[y, x]
            idx = int(val * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)
