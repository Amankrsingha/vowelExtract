# vowelpipe/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAV_DIR = ROOT / 'data' / 'raw' / 'wav'
TG_DIR = ROOT / 'data' / 'raw' / 'textgrids'
SPEAKER_CSV = ROOT / 'data' / 'metadata' / 'speakers.csv'
OUT_DIR = ROOT / 'output'

# ---- TextGrid layout (edit after running inspect_textgrids.py) ----
PHONE_TIER = 1          # 1-based position of the phone tier
WORD_TIER = 3           # 1-based position of the word tier, or None
# Vowel labels AFTER stress digits are removed (ARPABET shown here).
# If your TextGrids use IPA or another scheme, replace this set.
VOWELS = {'A', 'a', 'I', 'i', 'e', 'E', 'o', 'O', 'u'}
MIN_VOWEL_DUR = 0.001   # seconds; shorter vowels are skipped
ALIGN_TOLERANCE = 0.05  # seconds allowed between audio and TextGrid length

# ---- Formant analysis (Praat Burg method) ----
MAX_FORMANT = {'male': 5000.0, 'female': 5500.0}   # Hz, default ceilings
OPTIMISE_CEILING = False      # True: choose a ceiling per speaker (Section 4.2.3)
CEILING_CANDIDATES = [4500.0 + 250.0 * i for i in range(9)]   # 4500..6500 Hz
N_FORMANTS = 5.0
WINDOW = 0.025               # seconds
PREEMPH = 50.0               # Hz
TIME_STEP = None              # 0 = Praat default (a quarter of the window)
SAMPLE_POINTS = (0.2, 0.5, 0.8)   # proportions of vowel duration

# ---- f0 (used for descriptive purposes and quality control) ----
PITCH_RANGE = {'male': (60.0, 300.0), 'female': (100.0, 500.0)}

# ---- Quality control ----
F1_RANGE = (100.0, 5500.0)
F2_RANGE = (100.0, 5500.0)
OUTLIER_Z = 2.5
OUTLIER_MIN_N = 5
