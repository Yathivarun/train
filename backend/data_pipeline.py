"""
data_pipeline.py  —  Verified against real CSV data (Jan 2026 Howrah-Bandel section)

Two XGBoost dataset modes:
  1. build_xgboost_dataset()         — station-arrival based (1872 rows)
                                       Features: hour, day, prev_delay, train_type, station,
                                                 priority, station_avg_delay, num_tracks
                                       R² ≈ -0.16 (low signal — single day, no temporal spread)

  2. build_xgboost_section_dataset() — section-traffic based (1060 rows)  ← PREFERRED
                                       Features: excess_travel_time (corr=0.82!), hour,
                                                 train_type, stations, priority,
                                                 station_avg_delay, num_tracks,
                                                 distance_km, block_count
                                       R² ≈ 0.60, MAE ≈ 0.33 min

Correct source for train_type: section_train_master_expanded (all 812 trains)
NOT section_train_schedule_tuned (only 245/1872 rows have train_type)

Stations: HWH, BLY, SRP, CGR, CNS, BDC  (Howrah–Bandel corridor)
Train types: Local, MEMU, Passenger, Express
Priorities: 2=Local, 3=MEMU, 4=Passenger, 5=Express
"""

import os, logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data", "csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

STATION_ORDER = ["HWH", "BLY", "SRP", "CGR", "CNS", "BDC"]
TRAIN_TYPES   = ["Local", "MEMU", "Passenger", "Express"]


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        log.warning(f"Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    log.info(f"Loaded {filename}: {len(df)} rows")
    return df


# ─── Individual loaders ──────────────────────────────────────────────────────

def load_operational_event_log(): return _load("operational_event_log.csv")
def load_schedule():
    df = _load("section_train_schedule_tuned.csv")
    if not df.empty:
        df["scheduled_arrival_ts"] = pd.to_datetime(df["scheduled_arrival_ts"], errors="coerce")
        df["hour_of_day"] = df["scheduled_arrival_ts"].dt.hour
        df["day_of_week"] = df["scheduled_arrival_ts"].dt.dayofweek
    return df
def load_train_master():     return _load("section_train_master_expanded.csv")
def load_section_traffic():
    df = _load("section_traffic_final.csv")
    if not df.empty:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
        df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  errors="coerce")
        df["hour_of_day"] = df["entry_ts"].dt.hour
    return df
def load_delay_statistics():  return _load("delay_statistics_final.csv")
def load_section_master():    return _load("section_master_final.csv")
def load_edges():             return _load("section_edges.csv")
def load_kpi():               return _load("section_kpi_train.csv")
def load_baseline_timetable():return _load("baseline_timetable.csv")
def load_station_delay_log(): return _load("station_delay_log.csv")


# ─── PREFERRED: Section-traffic dataset (R²≈0.60, MAE≈0.33 min) ─────────────

def build_xgboost_section_dataset():
    """
    Uses section_traffic_final as base.
    Key insight: excess_travel_time = actual - min_run_time has 0.82 correlation
    with delay_minutes. This is the strongest available feature.

    Join chain:
      section_traffic_final (1060 rows)         ← base
        + section_edges                          ← excess_travel_time, distance_km, block_count
        + section_train_master_expanded          ← train_type, dispatch_priority
        + delay_statistics_final [PEAK/OFFPEAK]  ← station_avg_delay
        + section_master_final                   ← num_tracks

    Features (10): excess_travel_time, hour_of_day, train_type_enc,
                   from_station_enc, to_station_enc, dispatch_priority,
                   station_avg_delay, num_tracks, distance_km, block_count
    Target: delay_minutes (0–6.54 min, mean=0.809)
    """
    traffic   = load_section_traffic()
    edges     = load_edges()
    master    = load_train_master()
    stats     = load_delay_statistics()
    sec_master= load_section_master()

    if traffic.empty:
        log.error("section_traffic_final.csv missing")
        return None, None, None, None

    df = traffic.copy()

    # Excess travel time (primary feature, corr=0.82 with delay)
    if not edges.empty:
        df = df.merge(edges, on=["from_station","to_station"], how="left")
        df["excess_travel_time"] = (
            df["travel_time_min"] - df["min_run_time_min"]
        ).clip(lower=0)
    else:
        df["excess_travel_time"] = 0.0
        df["distance_km"]        = 8.0
        df["block_count"]        = 4

    # Train type and priority from master
    if not master.empty:
        df = df.merge(
            master[["train_no","train_type","dispatch_priority"]],
            on="train_no", how="left")
    else:
        df["train_type"]        = "Local"
        df["dispatch_priority"] = 2

    # Station capacity
    if not sec_master.empty:
        df = df.merge(
            sec_master[["station_code","num_tracks","peak_capacity_trains_per_hr"]],
            left_on="from_station", right_on="station_code", how="left")
    else:
        df["num_tracks"] = 4

    # Station avg delay by time window
    if not stats.empty:
        df["time_window"] = df["hour_of_day"].apply(
            lambda h: "PEAK" if (7<=h<=10 or 17<=h<=20) else "OFFPEAK")
        df = df.merge(
            stats[["station_code","time_window","avg_delay"]].rename(
                columns={"avg_delay":"station_avg_delay"}),
            left_on=["from_station","time_window"],
            right_on=["station_code","time_window"], how="left")
    else:
        df["station_avg_delay"] = 0.81

    # Encode
    df["train_type"]   = df["train_type"].fillna("Local").astype(str)
    le_train   = LabelEncoder().fit(TRAIN_TYPES)
    le_station = LabelEncoder().fit(STATION_ORDER)

    df["train_type_enc"]   = df["train_type"].apply(
        lambda x: int(le_train.transform([x])[0]) if x in le_train.classes_ else 0)
    df["from_station_enc"] = df["from_station"].apply(
        lambda x: int(le_station.transform([x])[0]) if x in le_station.classes_ else 0)
    df["to_station_enc"]   = df["to_station"].apply(
        lambda x: int(le_station.transform([x])[0]) if x in le_station.classes_ else 0)

    for col, val in [("dispatch_priority",2),("station_avg_delay",0.81),
                     ("num_tracks",4),("distance_km",8.0),("block_count",4)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

    FEATURE_COLS = [
        "excess_travel_time",   # corr=0.82 — primary signal
        "hour_of_day",
        "train_type_enc",
        "from_station_enc",
        "to_station_enc",
        "dispatch_priority",
        "station_avg_delay",
        "num_tracks",
        "distance_km",
        "block_count",
    ]
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["delay_minutes"].clip(lower=0)
    mask = y.notna()
    X, y = X[mask], y[mask]

    nulls = X.isnull().sum().sum()
    assert nulls == 0, f"NULLS IN X: {X.isnull().sum().to_dict()}"

    log.info(f"Section XGBoost dataset: {len(X)} samples | features={available}")
    log.info(f"  y: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.3f}")
    log.info(f"  excess_travel_time corr: {X['excess_travel_time'].corr(y):.4f}")

    joblib.dump(le_train,   os.path.join(MODELS_DIR, "le_train_type.pkl"))
    joblib.dump(le_station, os.path.join(MODELS_DIR, "le_station_code.pkl"))
    return X, y, le_train, le_station


# ─── FALLBACK: Station-arrival dataset (1872 rows, R²≈-0.16) ─────────────────

def build_xgboost_dataset():
    """
    Arrival-based dataset. Lower R² because single-day data has minimal
    temporal spread. Kept for per-station prediction endpoint compatibility.
    Delegates to section dataset if section data available.
    """
    # Prefer section dataset — better features, better R²
    X, y, le_t, le_s = build_xgboost_section_dataset()
    if X is not None:
        return X, y, le_t, le_s

    # True fallback: arrival-based
    ops    = load_operational_event_log()
    sched  = load_schedule()
    master = load_train_master()
    stats  = load_delay_statistics()
    sec_m  = load_section_master()

    if ops.empty:
        return None, None, None, None

    df = ops[ops["event_type"] == "ARRIVAL"].copy()

    if not sched.empty:
        df = df.merge(
            sched[["train_no","station_code","stop_sequence","hour_of_day","day_of_week"]],
            on=["train_no","station_code"], how="left")
    else:
        df["hour_of_day"]  = 0
        df["day_of_week"]  = 0
        df["stop_sequence"]= 1

    if not master.empty:
        df = df.merge(
            master[["train_no","train_type","dispatch_priority"]],
            on="train_no", how="left")
    else:
        df["train_type"]        = "Local"
        df["dispatch_priority"] = 2

    df = df.sort_values(["train_no","stop_sequence"])
    df["prev_delay"] = df.groupby("train_no")["delay_minutes"].shift(1).fillna(0.0)

    if not stats.empty:
        df["time_window"] = df["hour_of_day"].apply(
            lambda h: "PEAK" if (7<=h<=10 or 17<=h<=20) else "OFFPEAK")
        df = df.merge(
            stats[["station_code","time_window","avg_delay"]].rename(
                columns={"avg_delay":"station_avg_delay"}),
            on=["station_code","time_window"], how="left")
    else:
        df["station_avg_delay"] = 0.85

    if not sec_m.empty:
        df = df.merge(sec_m[["station_code","num_tracks"]], on="station_code", how="left")
    else:
        df["num_tracks"] = 4

    df["train_type"]   = df["train_type"].fillna("Local").astype(str)
    df["station_code"] = df["station_code"].fillna("HWH").astype(str)

    le_train   = LabelEncoder().fit(TRAIN_TYPES)
    le_station = LabelEncoder().fit(STATION_ORDER)
    df["train_type_enc"]   = df["train_type"].apply(
        lambda x: int(le_train.transform([x])[0]) if x in le_train.classes_ else 0)
    df["station_code_enc"] = df["station_code"].apply(
        lambda x: int(le_station.transform([x])[0]) if x in le_station.classes_ else 0)

    for col, val in [("hour_of_day",0),("day_of_week",0),("prev_delay",0.0),
                     ("dispatch_priority",2),("station_avg_delay",0.85),("num_tracks",4)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

    FEATURE_COLS = ["hour_of_day","day_of_week","prev_delay","train_type_enc",
                    "station_code_enc","dispatch_priority","station_avg_delay","num_tracks"]
    X = df[FEATURE_COLS].fillna(0)
    y = df["delay_minutes"].clip(lower=0)
    mask = y.notna()
    X, y = X[mask], y[mask]

    joblib.dump(le_train,   os.path.join(MODELS_DIR, "le_train_type.pkl"))
    joblib.dump(le_station, os.path.join(MODELS_DIR, "le_station_code.pkl"))
    return X, y, le_train, le_station


# ─── DQN replay dataset ───────────────────────────────────────────────────────

def build_dqn_replay_dataset():
    """
    State: [congestion_norm, active_trains_norm, priority_norm, time_of_day_norm]
    Source: section_traffic_final (1060 rows) + section_train_master_expanded
    Action label: 1 if delay_minutes > 2.0  (8.58% of rows)
    """
    traffic = load_section_traffic()
    master  = load_train_master()

    if traffic.empty:
        return pd.DataFrame()

    df = traffic.copy()

    if not master.empty:
        df = df.merge(master[["train_no","dispatch_priority"]], on="train_no", how="left")
        df["dispatch_priority"] = df["dispatch_priority"].fillna(2)
    else:
        df["dispatch_priority"] = 2

    max_delay = df["delay_minutes"].max()
    df["congestion_norm"] = (df["delay_minutes"] / max_delay).clip(0, 1)

    trains_per_hour = df.groupby("hour_of_day")["train_no"].count()
    max_per_hour    = trains_per_hour.max()
    df["active_trains_norm"] = df["hour_of_day"].map(trains_per_hour / max_per_hour).fillna(0.2)

    df["priority_norm"]    = (df["dispatch_priority"] / 5.0).clip(0, 1)
    df["time_of_day_norm"] = (df["hour_of_day"] / 23.0).clip(0, 1)
    df["action_label"]     = (df["delay_minutes"] > 2.0).astype(int)

    result = df[["congestion_norm","active_trains_norm","priority_norm",
                 "time_of_day_norm","delay_minutes","action_label"]].dropna().reset_index(drop=True)

    log.info(f"DQN replay: {len(result)} rows, action=1 rate={result['action_label'].mean():.2%}")
    return result


# ─── Metadata helpers ─────────────────────────────────────────────────────────

def get_station_list():      return STATION_ORDER
def get_train_types():       return TRAIN_TYPES
def get_priority_options():  return [2, 3, 4, 5]

def get_section_edges():
    edges = load_edges()
    return edges.to_dict("records") if not edges.empty else []

def get_train_schedule_lookup():
    sched = load_schedule()
    if sched.empty: return {}
    lookup = {}
    for t, g in sched.groupby("train_no"):
        cols = [c for c in ["station_code","stop_sequence","hour_of_day"] if c in g.columns]
        lookup[str(t)] = g[cols].to_dict("records")
    return lookup


if __name__ == "__main__":
    log.info("=== Data Pipeline Self-Test ===")

    log.info("--- Section dataset (preferred) ---")
    X, y, le_t, le_s = build_xgboost_section_dataset()
    if X is not None:
        log.info(f"Section XGBoost: X={X.shape}, nulls={X.isnull().sum().sum()} ✅")
        log.info(f"  Features: {list(X.columns)}")
    else:
        log.error("Section XGBoost failed ❌")

    log.info("--- DQN replay ---")
    dqn = build_dqn_replay_dataset()
    log.info(f"DQN: {dqn.shape} ✅" if not dqn.empty else "DQN: empty ⚠️")

    log.info(f"Stations: {get_station_list()}")
    log.info(f"Types: {get_train_types()}")
