# =============================================================================
#  sensor_simulator.py  —  PredictAI
#  Realistic live sensor simulation engine.
#  Feeds the Flask API with sensor readings from trained ML models.
#  FIXED: sklearn feature-name warnings fully suppressed.
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress ALL sklearn warnings globally at import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import joblib
import threading
import time
from datetime import datetime
from config import *

np.random.seed(RANDOM_SEED)

class SensorSimulator:
    """
    AR(1) sensor simulation per machine.
    Uses trained RF + GB + IF models to compute real failure probability.
    sklearn UserWarnings are fully suppressed — predictions use DataFrames.
    """
    FAULT_IMPACTS = {
        "overheat":          {"temperature":+30, "vibration":+0.5,  "current":+3.0},
        "bearing_wear":      {"vibration":+2.5,  "noise_level":+14, "temperature":+9},
        "pressure_leak":     {"pressure":-3.2,   "rpm":-200,        "current":+2.2},
        "electrical_fault":  {"voltage":-45,     "current":+6.0,    "noise_level":+10},
        "lubrication_loss":  {"oil_viscosity":-20,"temperature":+18, "vibration":+1.2},
    }

    def __init__(self):
        self.machines = {m["id"]: self._init_machine(m) for m in MACHINES}
        self.alerts   = []
        self.history  = {mid: [] for mid in self.machines}
        self._lock    = threading.Lock()
        self._model_ready = False
        self._load_models()

    def _init_machine(self, mdef):
        state = {"meta": mdef, "fault": None, "fault_intensity": 0.0,
                 "ticks": 0, "sensors": {},
                 "history": {s: [] for s in SENSOR_COLS}}
        for sensor, (mean, std) in SENSOR_NORMAL.items():
            offset = self._machine_offset(mdef["id"], sensor)
            state["sensors"][sensor] = float(np.random.normal(mean + offset, std * 0.3))
        return state

    @staticmethod
    def _machine_offset(mid, sensor):
        offsets = {
            "MCH-001": {"temperature":-2,  "vibration":-0.1, "rpm":+50},
            "MCH-002": {"temperature":-7,  "vibration":+0.3, "pressure":+6.2},
            "MCH-003": {"temperature":+6,  "vibration":+0.2, "rpm":+800},
            "MCH-004": {"temperature":-17, "vibration":-0.6, "pressure":-3.8, "rpm":-2350},
            "MCH-005": {"temperature":-4,  "vibration":-0.1, "pressure":+0.5, "rpm":-1000},
            "MCH-006": {"temperature":+8,  "vibration":+0.1, "rpm":+200},
        }
        return offsets.get(mid, {}).get(sensor, 0)

    def _load_models(self):
        try:
            self.rf           = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
            self.iso          = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
            self.gb           = joblib.load(os.path.join(MODEL_DIR, "gradient_boost.pkl"))
            self.scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            self.feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
            self._model_ready = True
            print("[SIM] Loaded: random_forest, gradient_boost, isolation_forest, scaler")
        except Exception as e:
            print(f"[SIM] Models not found — using rule-based fallback. ({e})")
            self._model_ready = False

    # ── Public: inject / clear faults ────────────────────────
    def inject_fault(self, machine_id, fault_type):
        with self._lock:
            if machine_id not in self.machines:
                return False
            m = self.machines[machine_id]
            m["fault"]           = fault_type
            m["fault_intensity"] = float(np.random.uniform(0.8, 1.6))
            self._log_alert("crit", machine_id, m["meta"]["name"],
                            f"FAULT INJECTED: {fault_type.replace('_',' ').upper()}")
            return True

    def clear_fault(self, machine_id):
        with self._lock:
            if machine_id in self.machines:
                m = self.machines[machine_id]
                m["fault"]           = None
                m["fault_intensity"] = 0.0
                self._log_alert("info", machine_id, m["meta"]["name"],
                                "Fault cleared — returning to normal")
                return True
        return False

    def clear_all_faults(self):
        for mid in list(self.machines.keys()):
            self.clear_fault(mid)

    # ── Tick: advance all machines one step ───────────────────
    def tick(self):
        hour = datetime.now().hour
        with self._lock:
            for mid, m in self.machines.items():
                self._update_sensors(m, hour)
                ml = self._ml_predict(m)
                m["ml"]    = ml
                m["ticks"] += 1

                # Full snapshot history (for /api/history)
                snap = {**m["sensors"],
                        "failProb": ml["failProb"],
                        "anomaly":  ml["anomaly"],
                        "health":   ml["health"],
                        "ts": datetime.now().strftime("%H:%M:%S")}
                self.history[mid].append(snap)
                if len(self.history[mid]) > 60:
                    self.history[mid].pop(0)

                # Per-sensor rolling history (for trend charts)
                for sensor in SENSOR_COLS:
                    m["history"][sensor].append(round(m["sensors"][sensor], 3))
                    if len(m["history"][sensor]) > 60:
                        m["history"][sensor].pop(0)

                # Also store anomaly + failProb history for sensor page charts
                for key in ("anomaly", "failProb"):
                    if key not in m["history"]:
                        m["history"][key] = []
                    m["history"][key].append(round(ml[key], 1))
                    if len(m["history"][key]) > 60:
                        m["history"][key].pop(0)

                # Auto-generate alerts
                st = ml["status"]
                if st == "crit" and np.random.random() < 0.12:
                    msg = f"{m['fault'].upper()} escalating" if m["fault"] \
                          else "Multi-sensor anomaly detected by ML"
                    self._log_alert("crit", mid, m["meta"]["name"], msg)
                elif st == "warn" and np.random.random() < 0.05:
                    self._log_alert("warn", mid, m["meta"]["name"],
                                    "Sensor values trending beyond threshold")

    def _update_sensors(self, m, hour):
        fi    = m["fault_intensity"]
        fault = m["fault"]
        load  = max(0.0, float(np.sin(np.pi * (hour - 6) / 12)))

        for sensor, (mean, std) in SENSOR_NORMAL.items():
            offset  = self._machine_offset(m["meta"]["id"], sensor)
            target  = mean + offset
            current = m["sensors"][sensor]
            noise   = np.random.normal(0, std * 0.07)
            m["sensors"][sensor] = 0.92 * current + 0.08 * target + noise

        # Daily load cycles
        m["sensors"]["temperature"] += load * 3.0
        m["sensors"]["current"]     += load * 1.2
        m["sensors"]["rpm"]         += load * 40

        # Fault escalation
        if fault and fault in self.FAULT_IMPACTS:
            for sensor, delta in self.FAULT_IMPACTS[fault].items():
                if sensor in m["sensors"]:
                    m["sensors"][sensor] += delta * fi * 0.15 + np.random.normal(0, abs(delta)*0.05)

        # Physical bounds
        bounds = {
            "temperature":   (20,  200),
            "vibration":     (0.0, 15.0),
            "pressure":      (0.0, 20.0),
            "rpm":           (50,  5000),
            "current":       (1.0, 40.0),
            "voltage":       (300, 480),
            "oil_viscosity": (5.0, 100.0),
            "noise_level":   (30,  120),
        }
        for s, (lo, hi) in bounds.items():
            m["sensors"][s] = float(np.clip(m["sensors"][s], lo, hi))

    def _ml_predict(self, m):
        """Compute real ML failure probability. Uses DataFrame → no sklearn warnings."""
        sensors = m["sensors"]

        # Rule-based baseline anomaly score
        total_anom = 0.0
        for col, lim in THRESHOLDS.items():
            if col in sensors:
                v = sensors[col]
                if col in ("voltage", "oil_viscosity"):
                    if v <= lim["crit"]:  total_anom += 0.30
                    elif v <= lim["warn"]: total_anom += 0.12
                else:
                    if v >= lim["crit"]:  total_anom += 0.30
                    elif v >= lim["warn"]: total_anom += 0.12
        anomaly_rule = min(total_anom, 1.0)

        if self._model_ready:
            try:
                row_scaled = self._build_feature_row(m)  # → numpy array
                # Wrap in DataFrame with correct feature names → silences warnings
                row_df = pd.DataFrame(row_scaled, columns=self.feature_cols)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rf_prob = float(self.rf.predict_proba(row_df)[:, 1][0])
                    gb_prob = float(self.gb.predict_proba(row_df)[:, 1][0])
                    iso_s   = float(-self.iso.decision_function(row_df)[0])

                iso_prob  = float(np.clip((iso_s + 0.5) / 1.5, 0.0, 1.0))
                fail_prob = 0.45*rf_prob + 0.30*gb_prob + 0.15*iso_prob + 0.10*anomaly_rule
                fail_prob = float(np.clip(fail_prob * 100, 1.0, 99.0))
                anomaly   = float(np.clip((rf_prob + anomaly_rule) / 2.0 * 100, 0.0, 100.0))
            except Exception:
                fail_prob = float(min(anomaly_rule * 80 + np.random.uniform(1, 8), 99))
                anomaly   = anomaly_rule * 100.0
        else:
            fail_prob = float(min(anomaly_rule * 80 + np.random.uniform(1, 8), 99))
            anomaly   = anomaly_rule * 100.0

        health = float(np.clip(100 - anomaly * 0.7 - np.random.uniform(0, 2), 5, 100))
        status = ("crit" if fail_prob > 65 or health < 35
                  else "warn" if fail_prob > 35 or health < 65
                  else "ok")

        return {
            "failProb": round(fail_prob, 1),
            "anomaly":  round(anomaly,   1),
            "health":   round(health,    1),
            "status":   status,
        }

    def _build_feature_row(self, m):
        """Build scaled feature vector matching the training feature set."""
        sensors = m["sensors"]
        hist    = m["history"]

        row = dict(sensors)

        # Rolling / lag features
        for col in ["temperature", "vibration", "pressure", "rpm"]:
            h = hist.get(col, [sensors[col]])
            row[f"{col}_roll5_mean"]  = float(np.mean(h[-5:]))  if h       else sensors[col]
            row[f"{col}_roll5_std"]   = float(np.std(h[-5:]))   if len(h)>1 else 0.0
            row[f"{col}_roll20_mean"] = float(np.mean(h[-20:])) if h       else sensors[col]
            row[f"{col}_delta"]       = float(h[-1] - h[-2])    if len(h)>=2 else 0.0
            row[f"{col}_lag1"]        = float(h[-2])             if len(h)>=2 else sensors[col]

        # Composite engineered features
        row["thermal_stress"]    = (sensors["temperature"] - 72) / 5.0
        row["mechanical_stress"] = sensors["vibration"] / 1.5
        row["hydraulic_stress"]  = sensors["pressure"]  / 6.0
        row["composite_stress"]  = (0.40 * row["thermal_stress"] +
                                    0.35 * row["mechanical_stress"] +
                                    0.25 * row["hydraulic_stress"])
        row["power_kw"]          = sensors["voltage"] * sensors["current"] / 1000.0
        row["rpm_deviation"]     = abs(sensors["rpm"] - 2800) / 2800.0
        row["lub_health"]        = (sensors["oil_viscosity"] - 25) / 30.0
        row["temp_vib_cross"]    = sensors["temperature"] * sensors["vibration"]
        row["temp_rpm_ratio"]    = sensors["temperature"] / max(sensors["rpm"], 1)
        row["vib_noise_cross"]   = sensors["vibration"]   * sensors["noise_level"]

        # Threshold violation count
        flags = 0
        for col, lim in THRESHOLDS.items():
            if col in sensors:
                v = sensors[col]
                if col in ("voltage", "oil_viscosity"):
                    if v <= lim["crit"]:  flags += 2
                    elif v <= lim["warn"]: flags += 1
                else:
                    if v >= lim["crit"]:  flags += 2
                    elif v >= lim["warn"]: flags += 1
        row["threshold_flags"] = flags

        # Cyclical time features
        h  = datetime.now().hour
        dw = datetime.now().weekday()
        row["hour_sin"] = float(np.sin(2 * np.pi * h  / 24))
        row["hour_cos"] = float(np.cos(2 * np.pi * h  / 24))
        row["day_sin"]  = float(np.sin(2 * np.pi * dw / 7))
        row["day_cos"]  = float(np.cos(2 * np.pi * dw / 7))

        # Machine one-hot encoding
        for col in self.feature_cols:
            if col.startswith("mch_") and col not in row:
                row[col] = 0
        mch_col = "mch_" + m["meta"]["id"]
        if mch_col in self.feature_cols:
            row[mch_col] = 1

        # Build DataFrame aligned to training columns → scale
        df_row = pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0)
        return self.scaler.transform(df_row)

    # ── Logging helper ────────────────────────────────────────
    def _log_alert(self, sev, mid, name, msg):
        self.alerts.insert(0, {
            "sev":  sev,
            "id":   mid,
            "name": name,
            "msg":  msg,
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        if len(self.alerts) > 100:
            self.alerts.pop()

    # ── Public getters for Flask API ──────────────────────────
    def get_state(self):
        with self._lock:
            result = {}
            for mid, m in self.machines.items():
                ml = m.get("ml", {"failProb": 5, "anomaly": 5, "health": 95, "status": "ok"})
                result[mid] = {
                    "meta":     m["meta"],
                    "sensors":  {k: round(v, 2) for k, v in m["sensors"].items()},
                    "fault":    m["fault"],
                    "failProb": ml["failProb"],
                    "anomaly":  ml["anomaly"],
                    "health":   ml["health"],
                    "status":   ml["status"],
                    "history":  {s: list(m["history"].get(s, [])) for s in list(SENSOR_COLS)+["anomaly","failProb"]},
                    "ticks":    m["ticks"],
                }
            return result

    def get_history(self, machine_id):
        with self._lock:
            return list(self.history.get(machine_id, []))

    def get_alerts(self, limit=50):
        with self._lock:
            return list(self.alerts[:limit])

    def get_fleet_kpis(self):
        with self._lock:
            ml_list  = [m.get("ml", {"failProb":5,"health":95,"status":"ok"})
                        for m in self.machines.values()]
            avg_h    = round(sum(x["health"]  for x in ml_list) / len(ml_list), 1)
            crits    = sum(1 for x in ml_list if x["status"] == "crit")
            warns    = sum(1 for x in ml_list if x["status"] == "warn")
            pred_f   = sum(1 for x in ml_list if x["failProb"] > 50)
            return {
                "avgHealth":        avg_h,
                "criticals":        crits,
                "warnings":         warns,
                "predictedFailures":pred_f,
                "online":           6,
                "total":            6,
            }