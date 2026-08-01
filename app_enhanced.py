"""
Solar Irradiance Forecasting + Energy Management System
=======================================================
Enhanced Gradio webapp built on Chronos-T5-base + LoRA fine-tuned forecaster.

New features over app.py:
  • ⚡ Battery Dispatch Simulator  (GHI → Solar Power → Battery → Grid KPIs)
  • 💰 Solar Savings Calculator    (₹ savings daily/monthly, CO₂ avoided)
  • 🌆 Multi-City Comparison       (side-by-side forecast + KPI table for 3 cities)
  • 📊 Energy Flow Visualization   (energy source breakdown chart)
  • 🔧 System Size Configurator    (panel kW + battery kWh + load profile)
  • 🌤️ Weather Overlay             (cloud cover, temperature, precipitation)

Run:  python app_enhanced.py   (launches on port 7861)
"""

import os, sys, json, threading, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import gradio as gr

# ─────────────────────────────────────────────
#  PROJECT PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR  = PROJECT_ROOT
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FT_RESULTS  = os.path.join(RESULTS_DIR, "unseen_finetuned")
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw")
UNSEEN_DIR  = os.path.join(PROJECT_ROOT, "data", "raw_unseen")

CONTEXT_LENGTH   = 72
FORECAST_HORIZON = 24

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# India grid emission factor (kg CO₂ per kWh grid electricity)
CO2_KG_PER_KWH = 0.82

# ─────────────────────────────────────────────
#  LAZY-LOADED GLOBALS
# ─────────────────────────────────────────────
_model_loaded = False
_pipeline_ft  = None
_device       = None

# ─────────────────────────────────────────────
#  CITY PRESETS
# ─────────────────────────────────────────────
QUICK_CITIES = {
    # ── North India ──
    "Delhi":                {"lat": 28.70, "lon":  77.10},
    "Noida":                {"lat": 28.54, "lon":  77.39},
    "Gurgaon":              {"lat": 28.46, "lon":  77.03},
    "Lucknow":              {"lat": 26.85, "lon":  80.95},
    "Jaipur":               {"lat": 26.91, "lon":  75.79},
    "Jodhpur":              {"lat": 26.24, "lon":  73.02},
    "Bikaner":              {"lat": 28.02, "lon":  73.32},
    "Udaipur":              {"lat": 24.58, "lon":  73.68},
    "Chandigarh":           {"lat": 30.73, "lon":  76.78},
    "Amritsar":             {"lat": 31.62, "lon":  74.87},
    "Shimla":               {"lat": 31.10, "lon":  77.17},
    "Dehradun":             {"lat": 30.32, "lon":  78.03},
    "Indore":               {"lat": 22.72, "lon":  75.86},
    "Bhopal":               {"lat": 23.26, "lon":  77.41},
    "Varanasi":             {"lat": 25.32, "lon":  82.97},
    # ── West India ──
    "Mumbai":               {"lat": 19.08, "lon":  72.88},
    "Pune":                 {"lat": 18.52, "lon":  73.86},
    "Nagpur":               {"lat": 21.15, "lon":  79.09},
    "Nashik":               {"lat": 20.00, "lon":  73.79},
    "Ahmedabad":            {"lat": 23.02, "lon":  72.57},
    "Surat":                {"lat": 21.20, "lon":  72.83},
    "Vadodara":             {"lat": 22.31, "lon":  73.18},
    "Rajkot":               {"lat": 22.30, "lon":  70.79},
    # ── South India ──
    "Bengaluru":            {"lat": 12.97, "lon":  77.59},
    "Chennai":              {"lat": 13.08, "lon":  80.27},
    "Hyderabad":            {"lat": 17.39, "lon":  78.49},
    "Coimbatore":           {"lat": 11.02, "lon":  76.97},
    "Madurai":              {"lat":  9.92, "lon":  78.12},
    "Tiruchirappalli":      {"lat": 10.79, "lon":  78.69},
    "Vijayawada":           {"lat": 16.51, "lon":  80.65},
    "Visakhapatnam":        {"lat": 17.69, "lon":  83.22},
    "Kochi":                {"lat":  9.93, "lon":  76.27},
    "Thiruvananthapuram":   {"lat":  8.52, "lon":  76.94},
    "Mysuru":               {"lat": 12.30, "lon":  76.64},
    "Mangaluru":            {"lat": 12.91, "lon":  74.85},
    # ── East India ──
    "Kolkata":              {"lat": 22.57, "lon":  88.36},
    "Patna":                {"lat": 25.59, "lon":  85.14},
    "Ranchi":               {"lat": 23.34, "lon":  85.31},
    "Bhubaneswar":          {"lat": 20.30, "lon":  85.82},
    "Guwahati":             {"lat": 26.14, "lon":  91.74},
    # ── Central India ──
    "Raipur":               {"lat": 21.25, "lon":  81.63},
    "Custom coordinates":   {"lat": None,  "lon":  None},
}

# ─────────────────────────────────────────────
#  LOAD PROFILES  (24-hour kW, hour 0 = midnight IST)
# ─────────────────────────────────────────────
_BASE_LOAD_PROFILES = {
    "Residential (3BHK)": np.array([
        0.4, 0.3, 0.3, 0.3, 0.4, 0.8,    # 00–05
        1.5, 2.0, 1.5, 1.0, 0.8, 0.8,    # 06–11
        0.9, 0.8, 0.9, 1.0, 1.2, 1.8,    # 12–17
        2.5, 2.8, 2.5, 2.0, 1.5, 0.8,    # 18–23
    ], dtype=np.float32),
    "Commercial Office": np.array([
        0.2, 0.2, 0.2, 0.2, 0.3, 0.5,
        1.0, 2.5, 3.5, 3.8, 3.8, 3.5,
        3.5, 3.8, 3.8, 3.5, 3.0, 2.0,
        1.0, 0.5, 0.3, 0.3, 0.2, 0.2,
    ], dtype=np.float32),
    "Industrial": np.array([
        1.0, 1.0, 1.0, 1.0, 1.5, 3.0,
        5.0, 6.0, 6.5, 6.5, 6.5, 6.0,
        6.5, 6.5, 6.5, 6.0, 5.5, 5.0,
        4.0, 3.0, 2.0, 1.5, 1.0, 1.0,
    ], dtype=np.float32),
}


def get_load_profile(profile_name: str, start_hour: int = 0) -> np.ndarray:
    """Return a 24-element load profile (kW) rotated to start at `start_hour` IST."""
    base = _BASE_LOAD_PROFILES.get(profile_name, _BASE_LOAD_PROFILES["Residential (3BHK)"])
    h    = int(start_hour) % 24
    return np.roll(base, -h).astype(np.float32)


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────

def fetch_recent_ghi(latitude, longitude, days_back=5):
    """Fetch recent hourly GHI + weather vars from Open-Meteo (ERA5, no API key)."""
    params = {
        "latitude":  latitude,
        "longitude": longitude,
        "hourly":    "shortwave_radiation,cloud_cover,temperature_2m,precipitation",
        "past_days": days_back,
        "timezone":  "UTC",
    }
    try:
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=45)
        except requests.exceptions.SSLError:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=45, verify=False)

        if resp.status_code != 200:
            return None, None, None, f"Open-Meteo HTTP {resp.status_code}."

        data       = resp.json()
        times      = data["hourly"]["time"]
        raw_ghi    = data["hourly"]["shortwave_radiation"]
        raw_cloud  = data["hourly"].get("cloud_cover",    [None] * len(times))
        raw_temp   = data["hourly"].get("temperature_2m", [None] * len(times))
        raw_precip = data["hourly"].get("precipitation",  [None] * len(times))

        if not times:
            return None, None, None, "Open-Meteo returned empty data."

        timestamps = pd.to_datetime(times, utc=True)
        ghi        = np.array([float(v) if v is not None else np.nan for v in raw_ghi],    dtype=np.float32)
        cloud      = np.array([float(v) if v is not None else np.nan for v in raw_cloud],  dtype=np.float32)
        temp       = np.array([float(v) if v is not None else np.nan for v in raw_temp],   dtype=np.float32)
        precip     = np.array([float(v) if v is not None else np.nan for v in raw_precip], dtype=np.float32)

        now_utc = pd.Timestamp.utcnow()
        cutoff  = now_utc.replace(minute=0, second=0, microsecond=0)
        mask    = timestamps <= cutoff

        if mask.sum() >= CONTEXT_LENGTH:
            ghi, cloud, temp, precip = ghi[mask], cloud[mask], temp[mask], precip[mask]
            timestamps = timestamps[mask]

        # Interpolate gaps
        def _clean(arr):
            s = pd.Series(arr).interpolate(method="linear", limit_direction="both").fillna(0)
            return s.values.astype(np.float32)

        ghi_clean = _clean(ghi)
        if len(ghi_clean) < CONTEXT_LENGTH:
            return None, None, None, f"Only {len(ghi_clean)}/{CONTEXT_LENGTH} hours available."

        weather = {
            "cloud_cover": _clean(cloud),
            "temperature": _clean(temp),
            "precipitation": _clean(precip),
        }
        start_str = times[0][:10]
        end_str   = times[mask.sum() - 1][:10] if mask.sum() >= CONTEXT_LENGTH else times[-1][:10]
        status    = f"Open-Meteo ERA5: {start_str} – {end_str} ({len(ghi_clean)} h)"
        return ghi_clean, timestamps.values, weather, status

    except requests.exceptions.Timeout:
        return None, None, None, "Open-Meteo API timed out."
    except Exception as e:
        return None, None, None, f"Fetch error: {e}"


def load_cached_city_data(city_key):
    """Load from pre-downloaded CSV when available."""
    city_lower = city_key.split(",")[0].strip().lower()
    for d in [UNSEEN_DIR, RAW_DIR]:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if city_lower in f.lower() and f.endswith(".csv"):
                df        = pd.read_csv(os.path.join(d, f), parse_dates=["timestamp"])
                df        = df.sort_values("timestamp")
                ghi       = df["ALLSKY_SFC_SW_DWN"].values.astype(np.float32)
                ts        = df["timestamp"].values
                ghi_clean = pd.Series(ghi).ffill().bfill().fillna(0).values.astype(np.float32)
                return ghi_clean, ts, None, f"Cached CSV: {len(ghi_clean)} hours"
    return None, None, None, None


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

def load_model():
    global _model_loaded, _pipeline_ft, _device
    if _model_loaded:
        return "✅ Model already loaded."

    import torch
    _device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " | CPU mode"

    try:
        from chronos import ChronosPipeline
        from peft import PeftModel

        lora_path = os.path.join(MODELS_DIR, "chronos_lora", "chronos_lora")

        _pipeline_ft = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map=_device,
            torch_dtype=torch.float16,
        )
        _pipeline_ft.model.model = PeftModel.from_pretrained(
            _pipeline_ft.model.model, lora_path
        ).to(_device)
        _pipeline_ft.model.model.eval()

        _model_loaded = True
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        return f"✅ Chronos-T5-base + LoRA loaded on {_device}{gpu_info} | VRAM: {vram:.2f} GB"
    except Exception as e:
        return f"❌ Load failed: {e}"


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

def _predict_native_multiscale(ghi_context, num_samples=20):
    """Weighted multi-scale ensemble: 72h / 48h / 24h contexts."""
    import torch
    scales = [(72, 0.5), (48, 0.3), (24, 0.2)]
    result  = np.zeros(FORECAST_HORIZON, dtype=np.float64)
    q10_out = q90_out = None

    for ctx_len, weight in scales:
        trimmed   = ghi_context[-ctx_len:]
        context_t = [torch.tensor(trimmed, dtype=torch.float32)]
        with torch.no_grad():
            forecast = _pipeline_ft.predict(
                context_t, prediction_length=FORECAST_HORIZON, num_samples=num_samples
            )
        samp = forecast.cpu().numpy()[0]
        med  = np.maximum(0, np.median(samp, axis=0))
        result += weight * np.clip(med, 0, 1400)
        if ctx_len == 72:
            q10_out = np.clip(np.percentile(samp, 10, axis=0), 0, 1400)
            q90_out = np.clip(np.percentile(samp, 90, axis=0), 0, 1400)

    return result.astype(np.float32), q10_out, q90_out


# ─────────────────────────────────────────────
#  ENERGY SIMULATION HELPERS
# ─────────────────────────────────────────────

def ghi_to_solar_power(ghi_arr: np.ndarray, panel_kw: float,
                        derating: float = 0.85, inverter: float = 0.96) -> np.ndarray:
    """Convert GHI (W/m²) → solar AC power output (kW)."""
    return (ghi_arr / 1000.0) * panel_kw * derating * inverter


def battery_dispatch(solar_kw, load_kw, capacity_kwh,
                     max_charge_kw, max_discharge_kw, initial_soc_frac=0.2):
    """
    Greedy hourly battery dispatch.
    Returns:
      grid_import (kW), grid_export (kW), batt_soc_kwh (kWh, len=25),
      solar_direct (kW), batt_discharge (kW), batt_charge (kW)
    """
    SOC_MIN = 0.10 * capacity_kwh
    SOC_MAX = 0.95 * capacity_kwh
    soc     = float(np.clip(initial_soc_frac * capacity_kwh, SOC_MIN, SOC_MAX))

    grid_import    = np.zeros(24, dtype=np.float32)
    grid_export    = np.zeros(24, dtype=np.float32)
    batt_soc_kwh   = np.zeros(25, dtype=np.float32)
    solar_direct   = np.zeros(24, dtype=np.float32)
    batt_discharge = np.zeros(24, dtype=np.float32)
    batt_charge    = np.zeros(24, dtype=np.float32)
    batt_soc_kwh[0] = soc

    for h in range(24):
        s   = float(solar_kw[h])
        l   = float(load_kw[h])
        net = s - l

        if net >= 0:                                         # surplus solar
            solar_direct[h] = l
            charge           = min(net, max_charge_kw, SOC_MAX - soc)
            charge           = max(0.0, charge)
            soc             += charge
            batt_charge[h]   = charge
            grid_export[h]   = max(0.0, net - charge)
        else:                                                # deficit
            solar_direct[h]   = s
            deficit            = -net
            discharge          = min(deficit, max_discharge_kw, soc - SOC_MIN)
            discharge          = max(0.0, discharge)
            soc               -= discharge
            batt_discharge[h]  = discharge
            grid_import[h]     = max(0.0, deficit - discharge)

        batt_soc_kwh[h + 1] = soc

    return grid_import, grid_export, batt_soc_kwh, solar_direct, batt_discharge, batt_charge


def compute_energy_kpis(solar_kw, load_kw, grid_import, grid_export,
                         batt_soc_kwh, batt_charge, batt_discharge, solar_direct,
                         capacity_kwh, panel_kw, electricity_rate):
    """Compute energy KPI dataframe."""
    solar_total     = float(np.sum(solar_kw))
    load_total      = float(np.sum(load_kw))
    grid_imp_total  = float(np.sum(grid_import))
    grid_exp_total  = float(np.sum(grid_export))
    batt_ch_total   = float(np.sum(batt_charge))
    batt_dis_total  = float(np.sum(batt_discharge))
    solar_dir_total = float(np.sum(solar_direct))

    solar_used      = solar_dir_total + batt_dis_total
    self_suff       = solar_used / load_total * 100 if load_total > 0 else 0.0
    self_cons       = solar_used / solar_total * 100 if solar_total > 0 else 0.0
    grid_reduction  = (load_total - grid_imp_total) / load_total * 100 if load_total > 0 else 0.0
    co2_avoided_kg  = solar_used * CO2_KG_PER_KWH
    daily_savings   = solar_used * electricity_rate
    monthly_savings = daily_savings * 30
    peak_reduction  = _peak_grid_reduction(solar_kw, load_kw, grid_import)

    rows = [
        ("☀  Total Solar Generated",  f"{solar_total:.2f} kWh"),
        ("🏠  Total Load",             f"{load_total:.2f} kWh"),
        ("⚡  Solar Used Directly",    f"{solar_dir_total:.2f} kWh"),
        ("🔋  Battery Charged",        f"{batt_ch_total:.2f} kWh"),
        ("🔋  Battery Discharged",     f"{batt_dis_total:.2f} kWh"),
        ("🔌  Grid Imported",          f"{grid_imp_total:.2f} kWh"),
        ("💚  Grid Exported",          f"{grid_exp_total:.2f} kWh"),
        ("📊  Self-Sufficiency",       f"{self_suff:.1f} %"),
        ("📊  Self-Consumption",       f"{self_cons:.1f} %"),
        ("📉  Grid Dependency Reduction", f"{grid_reduction:.1f} %"),
        ("🌿  CO₂ Avoided",           f"{co2_avoided_kg:.2f} kg"),
        ("💰  Daily Savings",         f"₹ {daily_savings:.2f}"),
        ("💰  Monthly Savings (est.)", f"₹ {monthly_savings:.2f}"),
        ("🕕  Evening Peak Reduction (6–10 PM)", f"{peak_reduction:.1f} %"),
    ]
    return pd.DataFrame(rows, columns=["KPI", "Value"])


def _peak_grid_reduction(solar_kw, load_kw, grid_import):
    """Compare grid import 6–10 PM (relative hours) vs baseline (no solar)."""
    eve_hours       = list(range(18, 22))  # 6 PM – 10 PM (0-indexed)
    baseline_import = sum(float(load_kw[h]) for h in eve_hours)
    actual_import   = sum(float(grid_import[h]) for h in eve_hours)
    if baseline_import <= 0:
        return 0.0
    return (baseline_import - actual_import) / baseline_import * 100


# ─────────────────────────────────────────────
#  ENERGY DASHBOARD PLOT  (4-panel)
# ─────────────────────────────────────────────

def plot_energy_dashboard(ghi_fc, q10, q90, solar_kw, solar_q10, solar_q90,
                           load_kw, grid_import, grid_export, batt_soc_kwh,
                           capacity_kwh, grid_baseline, ist_labels,
                           panel_kw, batt_kwh_cfg, city_name, lat, lon):
    hours  = np.arange(FORECAST_HORIZON)
    x_tick = list(range(0, 24, 3))
    x_lbl  = [ist_labels[i] for i in x_tick] if len(ist_labels) == 24 else [str(i) for i in x_tick]
    soc_pct = batt_soc_kwh[1:] / capacity_kwh * 100  # 24 values

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    dark_ax  = {"facecolor": "#1a1d23", "grid_alpha": 0.20}

    # ── Panel A: GHI Forecast + Solar Power ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(dark_ax["facecolor"])
    ax1.fill_between(hours, q10 / 1000 * panel_kw * 0.85 * 0.96,
                     q90 / 1000 * panel_kw * 0.85 * 0.96,
                     alpha=0.15, color="#FFD700")
    ax1.plot(hours, solar_kw,  color="#FFD700", lw=2.2, label="Solar Power (kW)")
    ax1.plot(hours, load_kw,   color="#FF6B6B", lw=2.0, ls="--", label="Load (kW)")
    ax1.fill_between(hours, solar_kw, load_kw,
                     where=(solar_kw >= load_kw), alpha=0.15, color="#2ecc71",
                     label="Surplus (charge)")
    ax1.fill_between(hours, solar_kw, load_kw,
                     where=(solar_kw < load_kw),  alpha=0.15, color="#e74c3c",
                     label="Deficit (discharge)")
    ax1.set_xticks(x_tick); ax1.set_xticklabels(x_lbl, fontsize=7, color="white")
    ax1.set_ylabel("Power (kW)", color="white")
    ax1.set_title(f"☀  Solar vs Load — {city_name}\n{panel_kw} kW panel • {batt_kwh_cfg} kWh battery",
                  color="white", fontweight="bold", fontsize=10)
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#444")
    ax1.legend(fontsize=7.5, loc="upper right", facecolor="#222", labelcolor="white",
               edgecolor="#555")
    ax1.set_ylim(bottom=0); ax1.grid(alpha=dark_ax["grid_alpha"], color="#aaa")

    # ── Panel B: Battery State of Charge ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(dark_ax["facecolor"])
    ax2.axhspan(10, 95, alpha=0.06, color="#2ecc71")
    ax2.axhline(95, color="#2ecc71", lw=0.8, ls="--", alpha=0.6, label="Max SOC (95%)")
    ax2.axhline(10, color="#e74c3c", lw=0.8, ls="--", alpha=0.6, label="Min SOC (10%)")

    # Colour segments: green = charging, orange = discharging
    for h in range(FORECAST_HORIZON):
        soc_start = batt_soc_kwh[h]   / capacity_kwh * 100
        soc_end   = batt_soc_kwh[h+1] / capacity_kwh * 100
        color     = "#2ecc71" if soc_end >= soc_start else "#FF9500"
        ax2.plot([h, h+1], [soc_start, soc_end], color=color, lw=2.5)

    ax2.fill_between(np.arange(25), batt_soc_kwh / capacity_kwh * 100,
                     alpha=0.12, color="#3498db",
                     step="post")
    ax2.set_xlim(0, 24); ax2.set_ylim(0, 105)
    ax2.set_xticks(x_tick); ax2.set_xticklabels(x_lbl, fontsize=7, color="white")
    ax2.set_ylabel("Battery SOC (%)", color="white")
    ax2.set_title("🔋  Battery State of Charge", color="white", fontweight="bold")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(fontsize=7.5, facecolor="#222", labelcolor="white", edgecolor="#555")
    ax2.grid(alpha=dark_ax["grid_alpha"], color="#aaa")

    # ── Panel C: Grid Import Profile ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(dark_ax["facecolor"])
    ax3.bar(hours, grid_baseline, color="#7f8c8d", alpha=0.45, label="Without Battery", width=0.7)
    ax3.bar(hours, grid_import,   color="#e74c3c", alpha=0.85, label="With Battery",    width=0.5)
    ax3.bar(hours, -grid_export,  color="#2ecc71", alpha=0.80, label="Grid Export (→)",  width=0.5)
    ax3.axhline(0, color="white", lw=0.5)
    # Highlight evening peak
    ax3.axvspan(17.5, 21.5, alpha=0.12, color="#f39c12", label="Evening peak (6–10 PM)")
    ax3.set_xticks(x_tick); ax3.set_xticklabels(x_lbl, fontsize=7, color="white")
    ax3.set_ylabel("kW  (negative = export)", color="white")
    ax3.set_title("🔌  Grid Import / Export", color="white", fontweight="bold")
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#444")
    ax3.legend(fontsize=7.5, facecolor="#222", labelcolor="white", edgecolor="#555")
    ax3.grid(alpha=dark_ax["grid_alpha"], color="#aaa")

    # ── Panel D: Hourly Energy Source Breakdown ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(dark_ax["facecolor"])

    # Stack: solar direct + battery discharge + grid import = load
    # (export is separate, handled above)
    solar_direct_h = np.minimum(solar_kw, load_kw)
    batt_dis_h     = np.minimum(np.maximum(load_kw - solar_direct_h, 0),
                                np.maximum(grid_baseline - grid_import, 0))
    grid_imp_h     = grid_import

    ax4.bar(hours, solar_direct_h, label="Solar Direct",    color="#FFD700", alpha=0.90, width=0.7)
    ax4.bar(hours, batt_dis_h,     bottom=solar_direct_h,   label="Battery",  color="#2ecc71", alpha=0.85, width=0.7)
    ax4.bar(hours, grid_imp_h,     bottom=solar_direct_h + batt_dis_h,
            label="Grid Import", color="#e74c3c", alpha=0.80, width=0.7)
    ax4.plot(hours, load_kw, "w--", lw=1.5, label="Total Load", alpha=0.9)

    ax4.set_xticks(x_tick); ax4.set_xticklabels(x_lbl, fontsize=7, color="white")
    ax4.set_ylabel("kW", color="white")
    ax4.set_title("📊  Hourly Energy Source Breakdown", color="white", fontweight="bold")
    ax4.tick_params(colors="white"); ax4.spines[:].set_color("#444")
    ax4.legend(fontsize=7.5, facecolor="#222", labelcolor="white", edgecolor="#555",
               loc="upper right")
    ax4.set_ylim(bottom=0); ax4.grid(alpha=dark_ax["grid_alpha"], color="#aaa")

    fig.suptitle(
        f"Energy Management Dashboard — {city_name}  ({lat:.2f}°N, {lon:.2f}°E)",
        color="white", fontweight="bold", fontsize=13, y=1.01
    )
    return fig


# ─────────────────────────────────────────────
#  ENERGY FLOW CHART
# ─────────────────────────────────────────────

def plot_energy_flow_chart(solar_total, solar_direct, batt_charged,
                            batt_discharged, grid_import, grid_export, load_total):
    """Horizontal sankey-style bar chart showing energy routing."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), facecolor="#0f1117")
    fig.subplots_adjust(hspace=0.50)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    # ── Top: Where does solar energy go? ──
    solar_other = max(0, solar_total - solar_direct - batt_charged - grid_export)
    vals_top    = [solar_direct, batt_charged, grid_export, solar_other]
    lbls_top    = ["Direct Use", "To Battery", "Grid Export", "Losses"]
    colors_top  = ["#FFD700", "#2ecc71", "#3498db", "#7f8c8d"]

    cum = 0
    for v, l, c in zip(vals_top, lbls_top, colors_top):
        if v <= 0:
            continue
        ax_top.barh(0, v, left=cum, height=0.5, color=c, alpha=0.88, label=l)
        if v > 0.05 * solar_total:
            ax_top.text(cum + v / 2, 0, f"{v:.2f} kWh\n({v/solar_total*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color="black", fontweight="bold")
        cum += v

    ax_top.set_xlim(0, solar_total * 1.02)
    ax_top.set_yticks([]); ax_top.set_xlabel("kWh", color="white")
    ax_top.set_title(f"☀  Solar Energy Routing  (Total: {solar_total:.2f} kWh)",
                     color="white", fontweight="bold")
    ax_top.legend(fontsize=8, loc="lower right", facecolor="#222",
                  labelcolor="white", edgecolor="#555")

    # ── Bottom: How is load satisfied? ──
    solar_used = solar_direct + batt_discharged
    grid_load  = max(0, load_total - solar_used)
    vals_bot   = [solar_direct, batt_discharged, grid_load]
    lbls_bot   = ["Solar Direct", "From Battery", "From Grid"]
    colors_bot = ["#FFD700", "#2ecc71", "#e74c3c"]

    cum = 0
    for v, l, c in zip(vals_bot, lbls_bot, colors_bot):
        if v <= 0:
            continue
        ax_bot.barh(0, v, left=cum, height=0.5, color=c, alpha=0.88, label=l)
        if v > 0.05 * load_total and load_total > 0:
            ax_bot.text(cum + v / 2, 0, f"{v:.2f} kWh\n({v/load_total*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color="black", fontweight="bold")
        cum += v

    ax_bot.set_xlim(0, load_total * 1.02)
    ax_bot.set_yticks([]); ax_bot.set_xlabel("kWh", color="white")
    ax_bot.set_title(f"🏠  Load Supply Sources  (Total Load: {load_total:.2f} kWh)",
                     color="white", fontweight="bold")
    ax_bot.legend(fontsize=8, loc="lower right", facecolor="#222",
                  labelcolor="white", edgecolor="#555")

    return fig


# ─────────────────────────────────────────────
#  WEATHER OVERLAY CHART
# ─────────────────────────────────────────────

def plot_weather_overlay(ghi_fc, weather_dict, ist_labels):
    """Plot GHI forecast with cloud cover / temperature / precipitation overlay."""
    if weather_dict is None:
        return None

    hours  = np.arange(FORECAST_HORIZON)
    x_tick = list(range(0, 24, 3))
    x_lbl  = [ist_labels[i] for i in x_tick] if len(ist_labels) == 24 else [str(i) for i in x_tick]

    cloud = weather_dict.get("cloud_cover", None)
    temp  = weather_dict.get("temperature",  None)
    prec  = weather_dict.get("precipitation", None)

    # Use last 24 hours of weather as proxy for forecast period
    if cloud is not None and len(cloud) >= 24:
        cloud = cloud[-24:]
    if temp is not None and len(temp) >= 24:
        temp = temp[-24:]
    if prec is not None and len(prec) >= 24:
        prec = prec[-24:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, facecolor="#0f1117")
    fig.subplots_adjust(hspace=0.35)

    def _style(ax, title):
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.set_title(title, color="white", fontweight="bold", fontsize=10)
        ax.set_xticks(x_tick); ax.set_xticklabels(x_lbl, fontsize=7, color="white")
        ax.grid(alpha=0.20, color="#aaa")

    # GHI
    axes[0].fill_between(hours, ghi_fc, alpha=0.3, color="#FFD700")
    axes[0].plot(hours, ghi_fc, color="#FFD700", lw=2)
    axes[0].set_ylabel("GHI (W/m²)", color="white")
    _style(axes[0], "☀  GHI Forecast (W/m²)")

    # Cloud cover
    if cloud is not None:
        axes[1].fill_between(hours, cloud, alpha=0.45, color="#95a5a6")
        axes[1].plot(hours, cloud, color="#bdc3c7", lw=1.8)
        axes[1].set_ylim(0, 105); axes[1].set_ylabel("Cloud Cover (%)", color="white")
    else:
        axes[1].text(12, 50, "No cloud data", ha="center", color="gray", fontsize=11)
    _style(axes[1], "🌥  Cloud Cover (%)")

    # Temperature
    if temp is not None:
        axes[2].fill_between(hours, temp, alpha=0.35, color="#e74c3c")
        axes[2].plot(hours, temp, color="#ff6b6b", lw=1.8)
        axes[2].set_ylabel("°C", color="white")
    else:
        axes[2].text(12, 25, "No temperature data", ha="center", color="gray", fontsize=11)
    _style(axes[2], "🌡  Temperature (°C)")

    fig.suptitle("Weather Conditions during Forecast Window", color="white",
                 fontweight="bold", fontsize=12)
    return fig


# ─────────────────────────────────────────────
#  COMBINED FORECAST + SIMULATE
# ─────────────────────────────────────────────

def _resolve_context(city_choice, custom_lat, custom_lon, window_offset, use_api):
    """Shared helper: fetch data, cut context window, build IST labels."""
    city_info = QUICK_CITIES.get(city_choice, {})
    lat   = custom_lat if city_choice == "Custom coordinates" else city_info.get("lat")
    lon   = custom_lon if city_choice == "Custom coordinates" else city_info.get("lon")
    if lat is None or lon is None:
        return None, None, None, None, None, None, None, "Enter lat/lon."

    ghi, timestamps, weather, status = None, None, None, None
    if not use_api:
        ghi, timestamps, weather, status = load_cached_city_data(city_choice)
    if ghi is None:
        ghi, timestamps, weather, status = fetch_recent_ghi(lat, lon, days_back=90)
    if ghi is None:
        return None, None, None, None, None, None, None, status

    total  = CONTEXT_LENGTH + FORECAST_HORIZON
    offset = max(0, min(int(window_offset), (len(ghi) - total) // total))
    if offset == 0:
        start = max(0, len(ghi) - CONTEXT_LENGTH)
    else:
        end   = len(ghi) - (offset - 1) * total
        start = max(0, end - total)

    context = ghi[start: start + CONTEXT_LENGTH]

    # IST labels
    _IST = timedelta(hours=5, minutes=30)
    start_hour_ist = 6
    ist_labels_flat  = [f"+{i+1}h" for i in range(FORECAST_HORIZON)]
    ist_labels_short = [f"+{i+1}h" for i in range(FORECAST_HORIZON)]

    if timestamps is not None and len(timestamps) >= start + CONTEXT_LENGTH:
        try:
            ctx_end = pd.Timestamp(timestamps[start + CONTEXT_LENGTH - 1])
            if ctx_end.tzinfo is None:
                ctx_end = ctx_end.tz_localize("UTC")
            first_fc = ctx_end.tz_convert("UTC").replace(tzinfo=None) + _IST + timedelta(hours=1)
            start_hour_ist = first_fc.hour
            ist_times = [first_fc + timedelta(hours=i) for i in range(FORECAST_HORIZON)]
            day0 = ist_times[0].date()
            ist_labels_flat  = [t.strftime("%H:%M IST") for t in ist_times]
            ist_labels_short = [
                t.strftime("%H:%M") if t.date() == day0 else t.strftime("%H:%M\n+1d")
                for t in ist_times
            ]
        except Exception:
            pass

    return lat, lon, context, weather, ist_labels_flat, ist_labels_short, start_hour_ist, status


def run_forecast_and_simulate(city_choice, custom_lat, custom_lon, window_offset, use_api,
                               panel_kw, batt_kwh, load_profile_name, electricity_rate):
    if not _model_loaded:
        return None, None, None, None, "❌ Load the model first (Forecast tab → Load Model)."

    lat, lon, context, weather, ist_flat, ist_short, start_hour, data_status = _resolve_context(
        city_choice, custom_lat, custom_lon, window_offset, use_api
    )
    if context is None:
        return None, None, None, None, data_status

    # Forecast
    ghi_fc, q10, q90 = _predict_native_multiscale(context)

    # Convert GHI → solar power
    solar_kw     = ghi_to_solar_power(ghi_fc, panel_kw)
    solar_kw_q10 = ghi_to_solar_power(q10,    panel_kw)
    solar_kw_q90 = ghi_to_solar_power(q90,    panel_kw)

    # Load profile aligned to forecast start hour
    load_kw = get_load_profile(load_profile_name, start_hour=start_hour)

    # Battery dispatch
    max_rate = batt_kwh / 2.0
    grid_import, grid_export, batt_soc_kwh, solar_direct, batt_dis, batt_chg = \
        battery_dispatch(solar_kw, load_kw, batt_kwh, max_rate, max_rate)

    # Baseline (no battery)
    grid_baseline = np.maximum(load_kw - solar_kw, 0).astype(np.float32)

    # KPIs
    kpi_df = compute_energy_kpis(
        solar_kw, load_kw, grid_import, grid_export, batt_soc_kwh,
        batt_chg, batt_dis, solar_direct, batt_kwh, panel_kw, float(electricity_rate)
    )

    city_name = city_choice.replace("Custom coordinates", f"Custom ({lat:.2f}°, {lon:.2f}°)")

    # Dashboard figure
    dashboard_fig = plot_energy_dashboard(
        ghi_fc, q10, q90, solar_kw, solar_kw_q10, solar_kw_q90,
        load_kw, grid_import, grid_export, batt_soc_kwh,
        batt_kwh, grid_baseline, ist_short,
        panel_kw, batt_kwh, city_name, lat, lon
    )

    # Flow chart
    flow_fig = plot_energy_flow_chart(
        solar_total    = float(np.sum(solar_kw)),
        solar_direct   = float(np.sum(solar_direct)),
        batt_charged   = float(np.sum(batt_chg)),
        batt_discharged= float(np.sum(batt_dis)),
        grid_import    = float(np.sum(grid_import)),
        grid_export    = float(np.sum(grid_export)),
        load_total     = float(np.sum(load_kw)),
    )

    # Weather overlay
    weather_fig = None
    if weather is not None:
        weather_fig = plot_weather_overlay(ghi_fc, weather, ist_flat)

    # Hourly table
    hourly_df = pd.DataFrame({
        "Time (IST)":       ist_flat,
        "GHI (W/m²)":      np.round(ghi_fc,          1),
        "Solar (kW)":       np.round(solar_kw,         3),
        "Load (kW)":        np.round(load_kw,           3),
        "Bat SOC (%)":      np.round(batt_soc_kwh[1:] / batt_kwh * 100, 1),
        "Grid Import (kW)": np.round(grid_import,       3),
        "Grid Export (kW)": np.round(grid_export,       3),
    })

    status_msg = (
        f"✅ {city_name}  ({lat:.2f}°N, {lon:.2f}°E)\n"
        f"Data: {data_status}\n"
        f"System: {panel_kw} kW panel | {batt_kwh} kWh battery | "
        f"{load_profile_name} | ₹{electricity_rate}/kWh"
    )
    return dashboard_fig, flow_fig, weather_fig, kpi_df, hourly_df, status_msg


# ─────────────────────────────────────────────
#  MULTI-CITY COMPARISON
# ─────────────────────────────────────────────

def run_multi_city_compare(city1, city2, city3, panel_kw, batt_kwh, load_profile_name,
                            electricity_rate, use_api):
    if not _model_loaded:
        return None, None, "❌ Load the model first."

    cities = [c for c in [city1, city2, city3] if c and c != "— none —"]
    if len(cities) < 2:
        return None, None, "Select at least 2 cities."

    results = []
    for city in cities:
        info = QUICK_CITIES.get(city, {})
        lat, lon = info.get("lat"), info.get("lon")
        if lat is None:
            continue

        lat_r, lon_r, context, weather, ist_flat, ist_short, start_hour, _ = _resolve_context(
            city, lat, lon, 0, use_api
        )
        if context is None:
            continue

        ghi_fc, q10, q90 = _predict_native_multiscale(context)
        solar_kw = ghi_to_solar_power(ghi_fc, panel_kw)
        load_kw  = get_load_profile(load_profile_name, start_hour=start_hour)
        max_rate = batt_kwh / 2.0
        grid_imp, grid_exp, batt_soc, sol_dir, batt_dis, batt_chg = \
            battery_dispatch(solar_kw, load_kw, batt_kwh, max_rate, max_rate)

        results.append({
            "city":     city,
            "lat":      lat,
            "lon":      lon,
            "ghi_fc":   ghi_fc,
            "q10":      q10,
            "q90":      q90,
            "solar_kw": solar_kw,
            "load_kw":  load_kw,
            "soc":      batt_soc,
            "grid_imp": grid_imp,
            "grid_exp": grid_exp,
            "sol_dir":  sol_dir,
            "batt_dis": batt_dis,
            "ist_short":ist_short,
        })

    if not results:
        return None, None, "No valid results — check city selection."

    # ── Comparison figure ──
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 12), facecolor="#0f1117")
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    if n == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]
    elif n == 2:
        axes = [[axes[0][0], axes[0][1]],
                [axes[1][0], axes[1][1]],
                [axes[2][0], axes[2][1]]]

    colors = ["#FFD700", "#2ecc71", "#3498db"]
    hours  = np.arange(FORECAST_HORIZON)

    for col, r in enumerate(results):
        city_lbl = r["city"]
        c        = colors[col % len(colors)]
        x_tick   = list(range(0, 24, 4))
        x_lbl    = [r["ist_short"][i] for i in x_tick] if len(r["ist_short"]) == 24 else [str(i) for i in x_tick]

        def _style_ax(ax, title):
            ax.set_facecolor("#1a1d23")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.set_title(title, color="white", fontsize=9, fontweight="bold")
            ax.set_xticks(x_tick); ax.set_xticklabels(x_lbl, fontsize=6.5, color="white")
            ax.grid(alpha=0.18, color="#aaa")

        # Row 0: GHI forecast
        ax = axes[0][col]
        ax.fill_between(hours, r["q10"], r["q90"], alpha=0.18, color=c)
        ax.plot(hours, r["ghi_fc"], color=c, lw=2.2)
        ax.set_ylabel("GHI (W/m²)", color="white")
        ax.set_ylim(bottom=0)
        _style_ax(ax, f"☀  GHI — {city_lbl}")

        # Row 1: Solar vs Load
        ax = axes[1][col]
        ax.plot(hours, r["solar_kw"], color=c,        lw=2.0, label="Solar")
        ax.plot(hours, r["load_kw"],  color="#FF6B6B", lw=1.8, ls="--", label="Load")
        ax.fill_between(hours, r["solar_kw"], r["load_kw"],
                         where=(r["solar_kw"] >= r["load_kw"]),
                         alpha=0.15, color="#2ecc71")
        ax.fill_between(hours, r["solar_kw"], r["load_kw"],
                         where=(r["solar_kw"] < r["load_kw"]),
                         alpha=0.15, color="#e74c3c")
        ax.set_ylabel("kW", color="white")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, facecolor="#222", labelcolor="white", edgecolor="#555")
        _style_ax(ax, f"⚡  Solar vs Load — {city_lbl}")

        # Row 2: Battery SOC
        ax = axes[2][col]
        soc_pct = r["soc"][1:] / batt_kwh * 100
        ax.fill_between(hours, soc_pct, alpha=0.25, color=c)
        ax.plot(hours, soc_pct, color=c, lw=2.0)
        ax.axhline(95, color="#2ecc71", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(10, color="#e74c3c", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(0, 105)
        ax.set_ylabel("SOC (%)", color="white")
        _style_ax(ax, f"🔋  Battery SOC — {city_lbl}")

    fig.suptitle(
        f"Multi-City Solar Comparison  |  {panel_kw} kW panel • {batt_kwh} kWh battery",
        color="white", fontweight="bold", fontsize=13
    )

    # ── KPI comparison table ──
    rows = []
    for r in results:
        grid_imp   = float(np.sum(r["grid_imp"]))
        solar_used = float(np.sum(r["sol_dir"]) + np.sum(r["batt_dis"]))
        load_total = float(np.sum(r["load_kw"]))
        solar_total= float(np.sum(r["solar_kw"]))
        co2_avoid  = solar_used * CO2_KG_PER_KWH
        daily_save = solar_used * float(electricity_rate)
        self_suff  = solar_used / load_total * 100 if load_total > 0 else 0
        rows.append({
            "City":                  r["city"],
            "Solar Generated (kWh)": round(solar_total, 2),
            "Grid Import (kWh)":     round(grid_imp,    2),
            "Self-Sufficiency (%)":  round(self_suff,   1),
            "CO₂ Avoided (kg)":      round(co2_avoid,   2),
            "Daily Savings (₹)":     round(daily_save,  2),
        })
    kpi_compare_df = pd.DataFrame(rows)

    status = f"Compared {len(results)} cities: " + ", ".join(r["city"] for r in results)
    return fig, kpi_compare_df, status


# ─────────────────────────────────────────────
#  ORIGINAL FORECAST FUNCTION (kept intact)
# ─────────────────────────────────────────────

def run_forecast(city_choice, custom_lat, custom_lon, window_offset, use_api):
    if not _model_loaded:
        return None, None, None, "Load the model first."

    city_info = QUICK_CITIES.get(city_choice, {})
    lat  = custom_lat if city_choice == "Custom coordinates" else city_info.get("lat")
    lon  = custom_lon if city_choice == "Custom coordinates" else city_info.get("lon")
    if lat is None or lon is None:
        return None, None, None, "Enter latitude and longitude."

    ghi, timestamps, weather, data_status = None, None, None, None
    if not use_api:
        ghi, timestamps, weather, data_status = load_cached_city_data(city_choice)
    if ghi is None:
        ghi, timestamps, weather, data_status = fetch_recent_ghi(lat, lon, days_back=90)
    if ghi is None:
        return None, None, None, data_status

    total       = CONTEXT_LENGTH + FORECAST_HORIZON
    n_hist      = max(0, (len(ghi) - total) // total)
    max_windows = 1 + n_hist
    window_offset = max(0, min(int(window_offset), max_windows - 1))

    if window_offset == 0:
        start     = max(0, len(ghi) - CONTEXT_LENGTH)
        has_truth = False
        actual    = None
    else:
        end       = len(ghi) - (window_offset - 1) * total
        start     = max(0, end - total)
        has_truth = start + total <= len(ghi)
        actual    = ghi[start + CONTEXT_LENGTH: start + total] if has_truth else None

    context = ghi[start: start + CONTEXT_LENGTH]

    _IST_OFFSET = timedelta(hours=5, minutes=30)
    ist_labels = ist_labels_flat = [f"+{i+1}h" for i in range(FORECAST_HORIZON)]
    fc_start_ist = ""
    if timestamps is not None and len(timestamps) >= start + CONTEXT_LENGTH:
        try:
            ctx_end = pd.Timestamp(timestamps[start + CONTEXT_LENGTH - 1])
            if ctx_end.tzinfo is None:
                ctx_end = ctx_end.tz_localize("UTC")
            ist_times = [
                ctx_end.tz_convert("UTC").replace(tzinfo=None) + _IST_OFFSET + timedelta(hours=i + 1)
                for i in range(FORECAST_HORIZON)
            ]
            day0 = ist_times[0].date()
            ist_labels = [
                t.strftime("%H:%M") if t.date() == day0 else t.strftime("%H:%M\n+1d")
                for t in ist_times
            ]
            ist_labels_flat = [t.strftime("%H:%M IST") for t in ist_times]
            fc_start_ist = ist_times[0].strftime("%d %b %H:%M IST")
        except Exception:
            pass

    nat_ms, q10, q90 = _predict_native_multiscale(context)
    ensemble         = nat_ms

    city_label = city_choice.replace("Custom coordinates", f"Custom ({lat:.2f}°, {lon:.2f}°)")
    h_ctx = np.arange(-CONTEXT_LENGTH, 0)
    h_fc  = np.arange(0, FORECAST_HORIZON)

    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    gs  = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.2], hspace=0.38, wspace=0.3)

    def _dark_ax(ax):
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    ax1 = fig.add_subplot(gs[0, :])
    _dark_ax(ax1)
    ax1.plot(h_ctx, context, color="#888", lw=1.5, alpha=0.7, label="Context (72 h ERA5)")
    ax1.axvline(0, color="white", lw=2, ls="--", alpha=0.5)
    ax1.axvspan(0, FORECAST_HORIZON, alpha=0.04, color="orange")
    if actual is not None:
        ax1.plot(h_fc, actual, "w-", lw=2.5, label="Actual", zorder=5)
    ax1.fill_between(h_fc, q10, q90, alpha=0.18, color="#2ecc71", label="80% CI")
    ax1.plot(h_fc, ensemble, "-", color="#2ecc71", lw=2.5, label="Chronos Forecast", zorder=4)
    ax1.set_xlabel("← Context (past 72 h) | Forecast (IST) →", color="white", fontsize=10)
    ax1.set_ylabel("GHI (W/m²)", color="white", fontsize=11)
    ax1.set_title(
        f"24-Hour GHI Forecast — {city_label}  |  Chronos-T5-base + LoRA",
        color="white", fontweight="bold", fontsize=12
    )
    ctx_tick_pos = [-72, -48, -24, 0]
    ctx_tick_lbl = ["-72h", "-48h", "-24h", "Now"]
    fc_idx       = list(range(2, FORECAST_HORIZON, 3))
    ax1.set_xticks(ctx_tick_pos + [h_fc[i] for i in fc_idx])
    ax1.set_xticklabels(ctx_tick_lbl + [ist_labels[i] for i in fc_idx],
                        fontsize=8, color="white")
    ax1.legend(fontsize=9, loc="upper left", ncol=2, facecolor="#222",
               labelcolor="white", edgecolor="#555")
    ax1.set_ylim(bottom=0); ax1.grid(alpha=0.2, color="#aaa")

    ax2 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax2)
    ax2.bar(h_fc, ensemble, 0.7, color="#2ecc71", alpha=0.85, label="Chronos Forecast")
    ax2.fill_between(h_fc, q10, q90, alpha=0.25, color="#2ecc71", step="mid")
    if actual is not None:
        ax2.step(h_fc, actual, "w-", lw=1.5, where="mid", label="Actual")
    ax2.set_xlabel("Forecast time (IST)", color="white")
    ax2.set_ylabel("GHI (W/m²)", color="white")
    ax2.set_title("Hourly Breakdown", color="white", fontweight="bold")
    ax2.set_xticks(h_fc[::3])
    ax2.set_xticklabels([ist_labels[i] for i in range(0, FORECAST_HORIZON, 3)],
                         fontsize=7, color="white")
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white", edgecolor="#555")
    ax2.grid(alpha=0.2, color="#aaa")

    ax3 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax3)
    if actual is not None:
        err = actual - ensemble
        colors_bar = ["#3498db" if e >= 0 else "#e74c3c" for e in err]
        ax3.bar(h_fc, err, 0.7, color=colors_bar, alpha=0.75)
        ax3.axhline(0, color="white", lw=1)
        mae_val = np.mean(np.abs(err))
        ax3.set_title(f"Residual  |  MAE = {mae_val:.1f} W/m²",
                      color="white", fontweight="bold")
        ax3.set_ylabel("Error (W/m²)", color="white")
    else:
        ax3.fill_between(h_fc, q10, q90, alpha=0.30, color="#2ecc71")
        ax3.plot(h_fc, ensemble, "-", color="#27ae60", lw=2, label="Forecast median")
        ax3.set_title("Chronos 80% Confidence Interval", color="white", fontweight="bold")
        ax3.set_ylabel("GHI (W/m²)", color="white")
        ax3.legend(fontsize=9, facecolor="#222", labelcolor="white", edgecolor="#555")
        ax3.set_ylim(bottom=0)
    ax3.set_xlabel("Forecast time (IST)", color="white")
    ax3.set_xticks(h_fc[::3])
    ax3.set_xticklabels([ist_labels[i] for i in range(0, FORECAST_HORIZON, 3)],
                         fontsize=7, color="white")
    ax3.grid(alpha=0.2, color="#aaa")

    plt.tight_layout()

    # Metrics
    if actual is not None:
        mae  = float(np.mean(np.abs(actual - ensemble)))
        rmse = float(np.sqrt(np.mean((actual - ensemble) ** 2)))
        ss   = np.sum((actual - ensemble) ** 2)
        st   = np.sum((actual - np.mean(actual)) ** 2)
        r2   = float(1 - ss / st) if st > 0 else 0
        mbe  = float(np.mean(ensemble - actual))
        denom = (np.abs(actual) + np.abs(ensemble)) / 2
        smape = float(np.mean(np.where(denom > 0, np.abs(actual - ensemble) / denom, 0)) * 100)
        metrics_df = pd.DataFrame([{
            "Model": "Chronos-T5 Fine-Tuned",
            "MAE (W/m²)": round(mae, 1),
            "RMSE": round(rmse, 1),
            "R²": round(r2, 4),
            "SMAPE (%)": round(smape, 1),
            "Bias (MBE)": round(mbe, 1),
        }])
    else:
        metrics_df = pd.DataFrame({"Note": ["Live forecast — no ground truth available."]})

    forecast_df = pd.DataFrame({
        "Time (IST)":   ist_labels_flat,
        "GHI Forecast": np.round(ensemble, 1),
        "Lower 80% CI": np.round(q10,      1),
        "Upper 80% CI": np.round(q90,      1),
    })

    energy   = np.sum(ensemble) / 1000
    peak_h   = int(np.argmax(ensemble))
    peak_val = float(ensemble.max())
    peak_ist = ist_labels_flat[peak_h]
    status_msg = (
        f"{city_label}  ({lat:.2f}°, {lon:.2f}°)\n"
        f"Data: {data_status}\n"
        f"Forecast starts: {fc_start_ist}\n"
        f"Predicted energy: {energy:.2f} kWh/m² over 24 h\n"
        f"Predicted peak: {peak_val:.0f} W/m² at {peak_ist}"
    )
    return fig, metrics_df, forecast_df, status_msg


def update_coords(city_choice):
    info      = QUICK_CITIES.get(city_choice, {})
    lat       = info.get("lat") or 20.0
    lon       = info.get("lon") or 78.0
    is_custom = city_choice == "Custom coordinates"
    return (
        gr.update(value=lat, interactive=is_custom),
        gr.update(value=lon, interactive=is_custom),
    )


# ─────────────────────────────────────────────
#  PRE-COMPUTED RESULTS
# ─────────────────────────────────────────────

def load_precomputed():
    data = {}
    for name, path in [
        ("averages", os.path.join(RESULTS_DIR, "model_averages.csv")),
        ("unseen",   os.path.join(FT_RESULTS,  "finetuned_unseen_results.csv")),
    ]:
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
    p = os.path.join(FT_RESULTS, "finetuned_summary.json")
    if os.path.exists(p):
        with open(p) as f:
            data["summary"] = json.load(f)
    return data

PRECOMPUTED = load_precomputed()


def plot_model_comparison():
    if "averages" not in PRECOMPUTED:
        return None
    df     = PRECOMPUTED["averages"]
    r2_col = next((c for c in ["R2", "R²"] if c in df.columns), df.columns[-1])
    colors = [
        "#e74c3c" if "Chronos" in str(m) else
        "#3498db" if m == "Transformer" else "#bdc3c7"
        for m in df["Model"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="#0f1117")
    for ax, col, title in zip(
        axes,
        ["MAE", "RMSE", r2_col],
        ["MAE (lower = better)", "RMSE (lower = better)", "R² (higher = better)"],
    ):
        ax.set_facecolor("#1a1d23"); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
        ax.barh(df["Model"], df[col], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel(col, color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelcolor="white")
        if col == r2_col:
            ax.axvline(0, color="white", lw=0.5, ls="--")
    plt.suptitle("Model Comparison — Bangalore Test Set (2022–2024)",
                 color="white", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def plot_unseen_heatmap():
    if "unseen" not in PRECOMPUTED:
        return None
    df         = PRECOMPUTED["unseen"]
    strategies = ["FT-Standard", "FT-MultiScale", "ZS-Standard", "ZS-MultiScale",
                  "Ensemble-FT+ZS", "Persistence"]
    cities     = df["City"].unique()
    heat = []
    for c in cities:
        row = []
        for s in strategies:
            sub = df[(df["City"] == c) & (df["Strategy"] == s)]
            row.append(sub["MAE"].values[0] if len(sub) > 0 else np.nan)
        heat.append(row)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d23"); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    sns.heatmap(
        np.array(heat), annot=True, fmt=".0f",
        xticklabels=["FT-Std", "FT-MS", "ZS-Std", "ZS-MS", "Ensemble", "Persist."],
        yticklabels=[c.capitalize() for c in cities],
        cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "MAE (W/m²)"},
    )
    ax.set_title("MAE by Strategy and City — held-out test set",
                 color="white", fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  GRADIO APP
# ─────────────────────────────────────────────

ABOUT_MD = """
## About the Enhanced Solar + Energy Management App

### Forecasting Engine
**Chronos-T5-base + LoRA** fine-tuned on 5 Indian cities (2013–2021).
Context: 72 h → Forecast: 24 h ahead (hourly GHI, W/m²).
Multi-scale ensemble: 72h / 48h / 24h weighted (0.50 / 0.30 / 0.20).

### Energy Simulation
| Step | Formula |
|------|---------|
| GHI → Solar Power | `P = (GHI/1000) × panel_kW × 0.85 × 0.96` |
| Battery dispatch | Greedy: charge surplus, discharge deficit |
| SOC bounds | Min 10 % · Max 95 % of capacity |
| CO₂ factor | 0.82 kg per kWh (India grid average) |

### Data Source
[Open-Meteo ERA5](https://open-meteo.com/) — free, no API key, hourly up to today.

### Limitations
- No cloud/NWP inputs to model — accuracy lower in monsoon/overcast.
- Trained on Indian latitudes (8–34°N).
- Battery model uses simplified greedy dispatch (no LP optimiser).
"""

def create_app():
    city_list = list(QUICK_CITIES.keys())

    css = """
    footer { display: none !important; }
    .gradio-container { background: #0f1117 !important; }
    """

    with gr.Blocks(
        title="Solar Forecasting + Energy Management",
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="emerald"),
        css=css,
    ) as app:

        gr.Markdown("""
        # Solar Forecasting + Energy Management System
        **Chronos-T5-base + LoRA · Battery Simulator · Multi-City Comparison · CO₂ & Savings KPIs**
        """)

        # ════════════════════════════════════════
        #  TAB 1 — GHI FORECAST
        # ════════════════════════════════════════
        with gr.Tab("☀ GHI Forecast"):
            with gr.Row():
                btn_load     = gr.Button("Load Model", variant="primary", scale=1)
                model_status = gr.Textbox(value="Not loaded", label="Model Status",
                                          interactive=False, scale=4)
            btn_load.click(fn=load_model, outputs=model_status)

            gr.Markdown("---")
            with gr.Row():
                city_dd1   = gr.Dropdown(choices=city_list, value="Delhi",
                                         label="City", scale=2)
                lat_in1    = gr.Number(value=28.70, label="Latitude",  precision=4, interactive=False)
                lon_in1    = gr.Number(value=77.10, label="Longitude", precision=4, interactive=False)
            city_dd1.change(fn=update_coords, inputs=city_dd1, outputs=[lat_in1, lon_in1])

            with gr.Row():
                win_dd1 = gr.Dropdown(
                    choices=[(f"Live (most recent 3 days)", 0),
                             ("~1 week ago",  2), ("~2 weeks ago", 4),
                             ("~1 month ago", 7), ("~2 months ago", 15)],
                    value=0, type="value", label="Context window"
                )
                use_api1 = gr.Checkbox(value=True, label="Fetch live from Open-Meteo")

            btn_fc1   = gr.Button("Generate 24-Hour Forecast", variant="primary", size="lg")
            status1   = gr.Textbox(label="Status", interactive=False)
            plot_fc1  = gr.Plot(label="Forecast")
            with gr.Row():
                metrics1  = gr.Dataframe(label="Accuracy Metrics")
                ftable1   = gr.Dataframe(label="24-Hour Forecast Table")

            btn_fc1.click(
                fn=run_forecast,
                inputs=[city_dd1, lat_in1, lon_in1, win_dd1, use_api1],
                outputs=[plot_fc1, metrics1, ftable1, status1],
            )

        # ════════════════════════════════════════
        #  TAB 2 — ENERGY SIMULATOR
        # ════════════════════════════════════════
        with gr.Tab("⚡ Energy Simulator"):
            gr.Markdown("""
            ### Solar + Battery Energy Simulation
            Runs the Chronos GHI forecast, converts it to solar AC power,
            simulates greedy battery dispatch, and shows energy KPIs.
            *Load the model first (☀ GHI Forecast tab).*
            """)

            with gr.Row():
                city_dd2 = gr.Dropdown(choices=city_list, value="Jaipur",
                                       label="City", scale=2)
                lat_in2  = gr.Number(value=26.91, label="Latitude",  precision=4, interactive=False)
                lon_in2  = gr.Number(value=75.79, label="Longitude", precision=4, interactive=False)
            city_dd2.change(fn=update_coords, inputs=city_dd2, outputs=[lat_in2, lon_in2])

            with gr.Row():
                win_dd2  = gr.Dropdown(
                    choices=[(f"Live (most recent 3 days)", 0),
                             ("~1 week ago",  2), ("~2 weeks ago", 4),
                             ("~1 month ago", 7), ("~2 months ago", 15)],
                    value=0, type="value", label="Context window"
                )
                use_api2 = gr.Checkbox(value=True, label="Fetch live from Open-Meteo")

            gr.Markdown("#### System Configuration")
            with gr.Row():
                panel_slider = gr.Dropdown(
                    choices=[1.0, 2.5, 5.0, 10.0], value=2.5,
                    label="Solar Panel Capacity (kW)",
                    info="Rated DC output of the solar array"
                )
                batt_slider  = gr.Dropdown(
                    choices=[5.0, 10.0, 15.0, 20.0], value=10.0,
                    label="Battery Capacity (kWh)",
                    info="Usable energy storage capacity"
                )
            with gr.Row():
                load_dd  = gr.Dropdown(
                    choices=list(_BASE_LOAD_PROFILES.keys()),
                    value="Residential (3BHK)",
                    label="Load Profile",
                    info="Typical hourly demand pattern"
                )
                rate_in  = gr.Number(value=7.0, label="Electricity Rate (₹/kWh)",
                                     info="Used to compute daily/monthly savings")

            btn_sim  = gr.Button("Run Forecast + Simulate", variant="primary", size="lg")
            status2  = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("#### Energy Dashboard")
            dash_plot = gr.Plot(label="Energy Dashboard (4-panel)")

            with gr.Row():
                flow_plot    = gr.Plot(label="Energy Flow Chart")
                weather_plot = gr.Plot(label="Weather Overlay")

            gr.Markdown("#### Energy KPIs")
            kpi_table = gr.Dataframe(label="KPIs", wrap=True)

            gr.Markdown("#### Hourly Breakdown Table")
            hourly_table = gr.Dataframe(label="Hourly Data")

            btn_sim.click(
                fn=run_forecast_and_simulate,
                inputs=[city_dd2, lat_in2, lon_in2, win_dd2, use_api2,
                        panel_slider, batt_slider, load_dd, rate_in],
                outputs=[dash_plot, flow_plot, weather_plot, kpi_table, hourly_table, status2],
            )

        # ════════════════════════════════════════
        #  TAB 3 — MULTI-CITY COMPARISON
        # ════════════════════════════════════════
        with gr.Tab("🌆 Multi-City Comparison"):
            gr.Markdown("""
            ### Compare Solar Forecasts Across 3 Cities
            Select up to 3 cities. The app runs Chronos forecasts for each,
            simulates the battery dispatch, and shows side-by-side charts + KPI table.
            *Load the model first (☀ GHI Forecast tab).*
            """)

            with gr.Row():
                comp_city1 = gr.Dropdown(choices=city_list, value="Jaipur",    label="City 1")
                comp_city2 = gr.Dropdown(choices=city_list, value="Chennai",   label="City 2")
                comp_city3 = gr.Dropdown(choices=["— none —"] + city_list,
                                         value="Bengaluru", label="City 3 (optional)")

            gr.Markdown("#### System (shared across all cities)")
            with gr.Row():
                comp_panel = gr.Dropdown(choices=[1.0, 2.5, 5.0, 10.0], value=2.5,
                                         label="Panel (kW)")
                comp_batt  = gr.Dropdown(choices=[5.0, 10.0, 15.0, 20.0], value=10.0,
                                         label="Battery (kWh)")
                comp_load  = gr.Dropdown(choices=list(_BASE_LOAD_PROFILES.keys()),
                                         value="Residential (3BHK)", label="Load Profile")
                comp_rate  = gr.Number(value=7.0, label="Rate (₹/kWh)")
                comp_api   = gr.Checkbox(value=True, label="Live API")

            btn_comp  = gr.Button("Compare Cities", variant="primary", size="lg")
            comp_status = gr.Textbox(label="Status", interactive=False)
            comp_plot   = gr.Plot(label="Multi-City Comparison")
            kpi_comp    = gr.Dataframe(label="KPI Comparison Table")

            btn_comp.click(
                fn=run_multi_city_compare,
                inputs=[comp_city1, comp_city2, comp_city3,
                        comp_panel, comp_batt, comp_load, comp_rate, comp_api],
                outputs=[comp_plot, kpi_comp, comp_status],
            )

        # ════════════════════════════════════════
        #  TAB 4 — RESULTS EXPLORER
        # ════════════════════════════════════════
        with gr.Tab("📈 Results Explorer"):
            gr.Markdown("## Pre-Computed Benchmark Results  (2022–2024 held-out test set)")
            with gr.Row():
                btn_comp_r    = gr.Button("All Models — Bangalore",         variant="primary")
                btn_heat_r    = gr.Button("MAE Heatmap — Held-out Cities",  variant="secondary")
            results_plot_r = gr.Plot()
            btn_comp_r.click(fn=plot_model_comparison, outputs=results_plot_r)
            btn_heat_r.click(fn=plot_unseen_heatmap,   outputs=results_plot_r)
            if "unseen" in PRECOMPUTED:
                gr.Dataframe(value=PRECOMPUTED["unseen"].round(2),
                             label="Full results table")

        # ════════════════════════════════════════
        #  TAB 5 — UPLOAD CSV
        # ════════════════════════════════════════
        with gr.Tab("📂 Upload CSV"):
            gr.Markdown("""
            ### Forecast from your own GHI data
            Upload a CSV with ≥ 72 rows of hourly GHI values.
            The app uses the **last 72 rows** as context → 24-hour forecast.
            """)
            with gr.Row():
                file_in  = gr.File(label="CSV file", file_types=[".csv"])
                col_in   = gr.Textbox(value="ALLSKY_SFC_SW_DWN", label="GHI column name")
            btn_up     = gr.Button("Forecast", variant="primary")
            up_status  = gr.Textbox(label="Status", interactive=False)
            up_plot    = gr.Plot()
            up_table   = gr.Dataframe(label="Forecast")

            def forecast_from_csv(file, col):
                if not _model_loaded:
                    return None, None, "Load model first."
                if file is None:
                    return None, None, "Upload a CSV."
                try:
                    df = pd.read_csv(file.name if hasattr(file, "name") else file)
                except Exception as e:
                    return None, None, f"CSV error: {e}"
                if col not in df.columns:
                    return None, None, f"Column '{col}' not found. Available: {', '.join(df.columns[:10])}"
                ghi = df[col].dropna().values.astype(np.float32)
                if len(ghi) < CONTEXT_LENGTH:
                    return None, None, f"Need {CONTEXT_LENGTH} rows, got {len(ghi)}."
                context          = ghi[-CONTEXT_LENGTH:]
                forecast, q10, q90 = _predict_native_multiscale(context)
                h_ctx = np.arange(-CONTEXT_LENGTH, 0)
                h_fc  = np.arange(0, FORECAST_HORIZON)
                fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
                ax.set_facecolor("#1a1d23"); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
                ax.plot(h_ctx, context, color="#888", lw=1.5, alpha=0.7, label="Context")
                ax.axvline(0, color="white", lw=2, ls="--", alpha=0.5)
                ax.fill_between(h_fc, q10, q90, alpha=0.18, color="#2ecc71", label="80% CI")
                ax.plot(h_fc, forecast, "-o", color="#2ecc71", lw=2, ms=4, label="Forecast")
                ax.set_xlabel("Hours relative to context end", color="white")
                ax.set_ylabel("GHI (W/m²)", color="white")
                ax.set_title("24-Hour Forecast — Uploaded CSV", color="white", fontweight="bold")
                ax.legend(fontsize=9, facecolor="#222", labelcolor="white", edgecolor="#555")
                ax.set_ylim(bottom=0); ax.grid(alpha=0.2, color="#aaa")
                plt.tight_layout()
                table = pd.DataFrame({
                    "Hour":          range(1, 25),
                    "GHI Forecast":  np.round(forecast, 1),
                    "Lower 80% CI":  np.round(q10,      1),
                    "Upper 80% CI":  np.round(q90,      1),
                })
                return fig, table, f"Forecast from {len(ghi)} uploaded rows."

            btn_up.click(fn=forecast_from_csv, inputs=[file_in, col_in],
                         outputs=[up_plot, up_table, up_status])

        # ════════════════════════════════════════
        #  TAB 6 — ABOUT
        # ════════════════════════════════════════
        with gr.Tab("ℹ About"):
            gr.Markdown(ABOUT_MD)

    return app


if __name__ == "__main__":
    _bg = threading.Thread(target=load_model, daemon=True)
    _bg.start()

    demo = create_app()
    print("\n" + "=" * 60)
    print("  Solar Forecasting + Energy Management System")
    print("  Open: http://localhost:7861")
    print("=" * 60 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
