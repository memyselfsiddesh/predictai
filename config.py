# =============================================================================
#  config.py  —  PredictAI  |  Shared configuration
# =============================================================================
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_SAMPLES     = 12_000
FAILURE_RATE  = 0.12
TEST_SIZE     = 0.20

# ── Sensors ──────────────────────────────────────────────────────────────────
SENSOR_COLS = ["temperature","vibration","pressure","rpm",
               "current","voltage","oil_viscosity","noise_level"]
TARGET_COL  = "failure"

# Normal operating: (mean, std)
SENSOR_NORMAL = {
    "temperature":   (72,  5.0),
    "vibration":     (1.5, 0.30),
    "pressure":      (6.0, 0.80),
    "rpm":           (2800, 120),
    "current":       (15.0, 1.5),
    "voltage":       (415,  8.0),
    "oil_viscosity": (46,   4.0),
    "noise_level":   (65,   5.0),
}

# ── Fault profiles (delta from normal) ───────────────────────────────────────
FAULT_PROFILES = {
    "overheat":          {"temperature":+28, "vibration":+0.4,  "current":+2.5},
    "bearing_wear":      {"vibration":+2.2,  "noise_level":+12, "temperature":+8},
    "pressure_leak":     {"pressure":-2.8,   "rpm":-180,        "current":+1.8},
    "electrical_fault":  {"voltage":-40,     "current":+5.0,    "noise_level":+8},
    "lubrication_loss":  {"oil_viscosity":-18,"temperature":+15, "vibration":+1.0},
}

# ── Warning / Critical thresholds ────────────────────────────────────────────
THRESHOLDS = {
    "temperature":   {"warn": 85,   "crit": 100},
    "vibration":     {"warn": 2.5,  "crit": 4.0},
    "pressure":      {"warn": 8.5,  "crit": 10.0},
    "rpm":           {"warn": 3200, "crit": 3500},
    "current":       {"warn": 18,   "crit": 22},
    "voltage":       {"warn": 390,  "crit": 370},
    "oil_viscosity": {"warn": 32,   "crit": 25},
    "noise_level":   {"warn": 80,   "crit": 90},
}

# ── ML hyperparameters ────────────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_split=4,
                 min_samples_leaf=2, class_weight="balanced",
                 random_state=RANDOM_SEED, n_jobs=-1)

IF_PARAMS  = dict(n_estimators=200, contamination=FAILURE_RATE,
                  random_state=RANDOM_SEED, n_jobs=-1)

SVM_PARAMS = dict(C=10, kernel="rbf", gamma="scale",
                  probability=True, class_weight="balanced")

FAILURE_PROB_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD    = 0.75

# ── Machine definitions ───────────────────────────────────────────────────────
MACHINES = [
    {"id":"MCH-001","name":"CNC Mill Alpha",    "type":"CNC Machine",       "nT":72, "nV":1.2,"nP":4.8, "nR":2800},
    {"id":"MCH-002","name":"Hydraulic Press B", "type":"Hydraulic Press",   "nT":65, "nV":2.1,"nP":12.4,"nR":1200},
    {"id":"MCH-003","name":"Compressor C",      "type":"Air Compressor",    "nT":78, "nV":1.8,"nP":8.2, "nR":3600},
    {"id":"MCH-004","name":"Conveyor Drive D",  "type":"Belt Conveyor",     "nT":55, "nV":0.8,"nP":2.1, "nR":450},
    {"id":"MCH-005","name":"Pump Station E",    "type":"Centrifugal Pump",  "nT":68, "nV":1.4,"nP":6.5, "nR":1800},
    {"id":"MCH-006","name":"Motor Drive F",     "type":"AC Induction Motor","nT":80, "nV":1.6,"nP":3.2, "nR":3000},
]
