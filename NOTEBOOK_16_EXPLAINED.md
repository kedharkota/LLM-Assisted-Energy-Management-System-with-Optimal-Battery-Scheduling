# Notebook 16 — Solar Battery Optimizer
## Complete Walkthrough: What, Why, and How

---

## The Big Picture — Why Does This Notebook Exist?

The first 15 notebooks in this project are all about **forecasting**: predicting how much sunlight (GHI — Global Horizontal Irradiance) will fall on 12 Indian cities 24 hours into the future using AI models.

But raw irradiance numbers in W/m² are not useful by themselves. An energy planner or homeowner needs to know:

- How many kilowatts of electricity will my solar panels actually produce?
- When should I charge or discharge my battery to avoid using expensive grid electricity?
- How much CO₂ am I actually saving compared to just using the grid?
- Does this work better in a desert city (Jaipur) vs. a monsoon city (Thiruvananthapuram)?

**Notebook 16 answers all of these questions.** It takes the AI's irradiance forecasts, simulates a real rooftop solar + battery system, and produces energy management KPIs for every Indian city in the study.

---

## Cell-by-Cell Breakdown

---

### Cell 1 — Markdown Title

Just a title cell explaining what this notebook does. No code.

---

### Cell 2 — Markdown Section Header

Section header: "1. Setup & Configuration". No code.

---

### Cell 3 — Setup & Configuration

**What it does:** Imports all libraries, sets paths, and defines three configuration dictionaries.

**Why each library:**

| Library | Why |
|---------|-----|
| `numpy` | Numerical array operations — all the hour-by-hour energy math |
| `pandas` | Loading and resampling the UCI load dataset |
| `matplotlib` | Drawing all 6 output charts |
| `json` | Reading city metadata (lat/lon/climate zone) |
| `os` | Building file paths that work on any computer |
| `warnings` | Suppressing harmless deprecation messages |

**The three configuration dictionaries defined here:**

#### PV_CONFIG — Solar Panel Settings

```python
PV_CONFIG = {
    'panel_peak_kw': 2.5,           # STC rated capacity of the rooftop system
    'system_derating': 0.85,        # Real-world loss factor
    'inverter_efficiency': 0.96,    # String inverter efficiency
    'min_irradiance_threshold': 50  # W/m² — below this, panels produce nothing
}
```

- **`panel_peak_kw = 2.5`** — A 2.5 kW rated solar system is a typical small Indian rooftop installation (around 8–10 panels). The "STC rating" (Standard Test Conditions) means this is what it produces at exactly 1000 W/m² irradiance, 25°C, with ideal wiring. Real life is never this perfect.
- **`system_derating = 0.85`** — Accounts for real-world losses: temperature (panels get hot and produce less), soiling (dust on panels is a big issue in Indian cities), cable resistance, and slight panel-to-panel mismatch. 15% total loss is a standard industry estimate.
- **`inverter_efficiency = 0.96`** — The inverter converts DC from the panels to AC for household use. Modern string inverters are ~96% efficient.
- **`min_irradiance_threshold = 50`** — Below 50 W/m² (early dawn, late dusk, heavy cloud), the inverter cannot operate and output is forced to zero.

**Why not `panel_efficiency = 0.18`?**

This is a key design decision. The STC rating (2.5 kW) already incorporates cell-level efficiency. The 18% cell efficiency is what was used to convert panel area to 2.5 kW in the first place. Multiplying by 0.18 again would halve the output incorrectly. The `system_derating` factor instead captures losses that happen *after* the panel — temperature drift, wiring, and dirt.

#### BATTERY_CONFIG — Battery Settings

```python
BATTERY_CONFIG = {
    'capacity_kwh': 5.0,            # Total energy storage
    'max_charge_kw': 2.5,           # Maximum charge rate
    'max_discharge_kw': 2.5,        # Maximum discharge rate
    'roundtrip_efficiency': 0.92,   # 92% round-trip efficiency
    'charge_efficiency': sqrt(0.92),    # ~95.9% per direction
    'discharge_efficiency': sqrt(0.92), # ~95.9% per direction
    'min_soc': 0.10,                # Never go below 10%
    'max_soc': 0.95,                # Never go above 95%
    'initial_soc': 0.20,            # Start the day at 20%
}
```

- **`capacity_kwh = 5.0`** — A 5 kWh lithium iron phosphate (LFP) battery. This is representative of Indian home storage products (similar to the BYD Blade, Luminous Li-ON, etc.).
- **`max_charge_kw = 2.5`** — The battery charges at a maximum of 2.5 kW. This is called the **C-rate**. A 0.5C rate (2.5 kW for a 5 kWh battery) is considered safe and extends battery life. Charging too fast generates heat and degrades the cells.
- **`roundtrip_efficiency = 0.92`** — When you put 1 kWh in and take it out again, you only get 0.92 kWh back. The other 8% is lost as heat. To split this symmetrically: √0.92 ≈ 0.959 (95.9%) loss each way.
- **`min_soc = 0.10`** — Protective floor. Discharging below 10% damages lithium cells and shortens battery life. So only 85% of capacity (0.5 to 4.75 kWh) is actually usable.
- **`max_soc = 0.95`** — Keeping the top 5% empty reduces cell stress and extends calendar life.
- **`initial_soc = 0.20`** — The simulation starts the day at 20% to represent a conservative morning state (battery partially depleted from overnight use).

#### GRID_CONFIG — Electricity Grid Settings

```python
GRID_CONFIG = {
    'india_emission_factor': 0.82,  # kg CO₂ per kWh
    'peak_hours': [18, 19, 20, 21]  # 6PM–10PM
}
```

- **`india_emission_factor = 0.82`** — Every kWh drawn from the Indian grid produces ~0.82 kg of CO₂. This is the CEA (Central Electricity Authority) 2023–24 figure. India's grid is heavily coal-dependent (~70% thermal), making it one of the most carbon-intensive grids in the world. This is why solar adoption here has outsized climate impact.
- **`peak_hours`** — 6 PM to 10 PM is when Indian households collectively draw the most power (lights, fans, TVs, cooking). Grid stress peaks at this time and electricity is most expensive.

**`LOAD_SCALE_FACTOR = 0.50`** — The UCI dataset is from a French household consuming ~26 kWh/day. Indian middle-class urban households consume 12–15 kWh/day. Multiplying by 0.5 maps the French demand to a realistic Indian profile.

---

### Cell 4 — Markdown: "2. Load Irradiance Predictions"

Section header. No code.

---

### Cell 5 — Load All Predictions

**What it does:** Reads `.npy` files (NumPy binary arrays) from the `results/` folder for all 12 Indian cities and also loads the ground truth test targets.

**Key terms:**

- **`predictions[city]`** — A 2D NumPy array of shape **(729, 24)**. This means: 729 test days, each with 24 hourly GHI values in W/m². These came from the Chronos fine-tuned model run in Notebook 08.
- **`ground_truth[city]`** — The actual measured GHI values for those same 729 days. Shape (729, 24). These are what really happened.
- **Why Chronos fine-tuned specifically?** — In Notebook 11, the evaluation showed that LoRA-fine-tuned Chronos was the best-performing model across all 12 cities and all 5 climate zones. So this notebook uses only those predictions.

**What the output looks like:**
```
City                   Pred Shape       Target Shape     Pred Mean W/m²   Pred Max W/m²
ahmedabad              (729, 24)        (729, 24)        175.3            1018.2
bangalore              (729, 24)        (729, 24)        163.4            871.6
...
```

---

### Cell 6 — Markdown: "3. Load & Prepare Real Household Load Profile"

Section header. No code.

---

### Cell 7 — Load UCI Household Dataset

**What it does:** Reads the UCI electricity consumption dataset, resamples from per-minute to per-hour, scales down to Indian levels, and extracts a library of 1,417 complete 24-hour load profiles.

**Why real household data instead of synthetic?**

Synthetic load profiles look smooth and predictable. Real household data has:
- Random spikes when the washing machine turns on
- Irregular morning and evening peaks
- Day-to-day variability (weekends vs. weekdays, seasonal behavior)

This variability is important for testing how well the battery dispatch handles real-world unpredictability.

**Processing steps explained:**

| Step | Code | Why |
|------|------|-----|
| Read with `;` separator | `sep=';'` | The file uses semicolons not commas |
| Parse datetime | `parse_dates={'datetime': ['Date','Time']}` | Combine two columns into one proper timestamp |
| `dayfirst=True` | Date format is DD/MM/YYYY (European) | Prevents month/day confusion |
| `na_values='?'` | Missing readings are stored as `?` | Converts them to NaN so they can be dropped |
| `resample('1H').mean()` | 60 minute-level readings → 1 hourly mean | Reduces noise, matches solar data resolution |
| `× 0.50` | Scale factor | Converts French (~26 kWh/day) to Indian (~13 kWh/day) |

**What is `daily_profiles`?**

After resampling, the code loops over each calendar date and extracts a full 24-value array (one per hour). Only complete days (all 24 hours present, no NaN) are kept. The result is a 2D array of shape **(1417, 24)** — 1,417 real household days, each with 24 hourly power readings in kW (Indian-scaled).

---

### Cell 8 — Visualize Load Profile

**What it does:** Produces two side-by-side plots and saves `load_profile_overview.png`.

- **Left plot:** The mean hourly profile (average of all 1,417 days) with a grey ±1 standard deviation band. This shows the typical shape: low overnight, rising in the morning, a broad midday plateau, and a sharp evening peak.
- **Right plot:** 20 randomly sampled individual days (thin blue lines) overlaid with the mean (thick red). This shows how variable real consumption is — some days are spiky, some smooth.

It also prints a text bar chart (using `█` characters) of the hourly average, so you can see the load shape directly in the notebook output.

---

### Cell 9 — Markdown: PV Conversion & Battery Dispatch Engine

Section header with mathematical formulas. No code.

---

### Cell 10 — Core Functions (The Heart of the Notebook)

**What it does:** Defines three Python functions that do all the heavy lifting.

---

#### `irradiance_to_solar_kw(ghi_wm2, config)`

**Input:** Array of 24 GHI values in W/m²  
**Output:** Array of 24 solar power values in kW

**The formula:**
```
solar_kW[h] = (GHI[h] / 1000) × 2.5 × 0.85 × 0.96
```

- `GHI / 1000` converts from W/m² to a fraction of STC irradiance (STC is defined at exactly 1000 W/m²)
- Multiplying by `2.5` applies the panel's STC power rating
- Multiplying by `0.85` applies system derating (temperature, soiling, etc.)
- Multiplying by `0.96` applies inverter efficiency
- Any hour where GHI < 50 W/m² → output forced to 0 (inverter cutoff)

**Example:**  
GHI = 800 W/m² → `(800/1000) × 2.5 × 0.85 × 0.96 = 1.63 kW`  
GHI = 30 W/m² (dawn) → `0.0 kW` (below threshold)

---

#### `simulate_battery_dispatch(solar_kw, load_kw, config)`

This is the most complex function. It simulates what happens to a home battery every hour of the day.

**Inputs:**
- `solar_kw` — 24-element array of solar PV output (kW)
- `load_kw` — 24-element array of household consumption (kW)

**Output:** A dictionary with 24-element arrays for every energy flow, plus SOC at every hour.

**Key concept — `net`:**
```python
net = solar_kw[h] - load_kw[h]
```
This is the net power balance at each hour:
- `net > 0` = solar surplus (generating more than consuming)
- `net < 0` = solar deficit (consuming more than generating)

**The 3-hour look-ahead logic:**
```python
future_deficit = sum of max(0, load[h+j] - solar[h+j]) for j = 1, 2, 3
if future_deficit > 1.0 AND SOC < 70% AND net > 0:
    pre_charge 0.5 kW now
```
This is a simple form of **predictive control**. At noon, the solar might be strong and the battery full — but the model looks 3 hours ahead. If it sees that evenings will be heavily load-dominant, it tries to keep more charge in the battery rather than exporting to grid.

**Surplus hour (net > 0):**
1. All load is met directly by solar (`solar_direct = load`)
2. The leftover solar goes to charge the battery (up to max_charge_kw and max_SOC)
3. Anything remaining is exported to the grid

**Deficit hour (net < 0):**
1. All solar is used directly (`solar_direct = solar_kw`)
2. The deficit is filled first by the battery (up to max_discharge_kw and min_SOC)
3. Whatever the battery can't cover is imported from the grid

**SOC update:**
- Charging: `new_SOC = old_SOC + charge_kw × charge_efficiency` (95.9% stored)
- Discharging: `new_SOC = old_SOC - discharge_kw / discharge_efficiency` (the battery gives out 95.9% of what it drew from the cells)
- After every hour, SOC is clamped to the [10%, 95%] range

**Return dictionary keys:**

| Key | Meaning | Unit |
|-----|---------|------|
| `solar_kw` | Solar generation | kW per hour |
| `load_kw` | Household consumption | kW per hour |
| `soc_pct` | Battery state-of-charge at *start* of each hour | % |
| `soc_pct_end` | Battery state-of-charge at *end* of each hour | % |
| `charge_kw` | Power flowing into battery | kW per hour |
| `discharge_kw` | Power flowing out of battery | kW per hour |
| `grid_import_kw` | Power drawn from the grid | kW per hour |
| `grid_export_kw` | Power pushed back to the grid | kW per hour |
| `solar_direct_kw` | Solar power used directly (no battery) | kW per hour |

---

#### `compute_kpis(result, emission_factor)`

Takes the dispatch result dictionary and computes 11 summary metrics.

**The 4 main KPIs explained:**

| KPI | Formula | What it means |
|-----|---------|---------------|
| **Self-Sufficiency %** | `(1 - total_grid_import / total_load) × 100` | Out of every kWh consumed, what fraction did NOT come from the grid? A 90% self-sufficient home only needed 10% grid power for the day. |
| **CO₂ Avoided (kg)** | `(total_load - total_grid_import) × 0.82` | Every kWh that came from solar/battery instead of the grid avoided 0.82 kg of CO₂. This is the climate benefit. |
| **Peak Hours Avoided** | Count of hours where `grid_import < 0.1 kW AND load > 0.3 kW` | How many hours was the household genuinely grid-independent while actively consuming power? This is a count per day, out of 24. |
| **Solar Self-Consumption %** | `(solar_direct + battery_charge) / total_solar × 100` | Out of all the solar generated, how much was used on-site vs. wasted by exporting? High export means the battery is too small relative to the panel. |

**The 7 supporting KPIs:**
- `total_load_kwh` — Day's total electricity consumption
- `total_solar_kwh` — Day's total solar generation
- `total_grid_import_kwh` — How much came from the grid
- `total_grid_export_kwh` — How much was sent back to the grid
- `max_soc_pct` / `min_soc_pct` — Battery swing range
- `battery_cycles` — Full equivalent charge cycles used (e.g., 0.5 = half a cycle)

---

### Cell 11 — Markdown: "5. Single-City Simulation"

Section header. No code.

---

### Cell 12 — Single-City Demo (Jaipur)

**What it does:** Runs the full pipeline for Jaipur using one selected day.

**Why Jaipur?**  
Jaipur has the highest annual average irradiance (570 W/m²) in the dataset. It's a hot desert climate (Köppen BWh) — very sunny, very dry, ideal for solar.

**Day selection logic:**

```python
# Sort all 729 test days by total daily irradiance, pick the 75th percentile
p75_idx = np.argsort(daily_totals)[int(0.75 * len(daily_totals))]
```

Why 75th percentile and not the maximum?  
Using the absolute sunniest day would be misleading — it's an outlier. The 75th percentile represents "a good but realistic sunny day" — sunny enough to show the system working well, but not an extreme outlier.

**Load day selection:**

```python
median_idx = np.argsort(daily_consumption)[len(daily_consumption) // 2]
```

Similarly, a median-consumption load day avoids showing an atypically high or low demand day. This gives a realistic picture of daily operation.

**Typical output for Jaipur:**
```
Total Solar Generation:  13.87 kWh
Total Household Load:    12.93 kWh
Grid Import:             1.58 kWh
Grid Export:             2.32 kWh
Self-Sufficiency:        87.8%
Solar Self-Consumption:  94.1%
CO₂ Avoided:            9.30 kg
Peak Hours Avoided:      18
Battery Cycles:          0.685
SOC Range:               10.2% – 94.7%
```

---

### Cell 13 — Chart 1: Energy Flows

**What it shows:** A full 24-hour stacked view of where energy is coming from and going to.

- **Gold area** — Solar generation (peaks at solar noon, zero at night)
- **Red dashed line** — Household load (the demand curve)
- **Green bars** — Battery discharge (battery helping during deficit hours)
- **Red fill** — Grid import (what could not be covered by solar + battery)
- **Blue fill** — Grid export (surplus solar pushed to grid when battery is full)

**How to read it:**  
When the gold solar area is above the red demand line, it's a surplus period. When demand is above solar, the green battery bars show how much the battery is contributing. Narrow red fill means the battery covered most of the deficit.

Saved as `chart1_energy_flows_jaipur.png`.

---

### Cell 14 — Chart 2: Battery SOC & Charge/Discharge

**What it shows:** Two panels:

- **Top:** Battery state-of-charge (%) across 24 hours. You can see the battery charging during the sunny midday hours (SOC rising) and discharging in the evening (SOC falling). The 10% floor and 95% ceiling are visible as hard boundaries.
- **Bottom:** Bar chart showing charge (green, upward) and discharge (red, downward) at each hour. Positive bars = battery being filled. Negative bars = battery being depleted.

**Why this chart matters:**  
The SOC trajectory reveals whether the battery is being used efficiently. A battery that swings from 20% to 94% is being fully utilized. One that only moves between 40% and 60% suggests the panel is undersized.

Saved as `chart2_battery_soc_jaipur.png`.

---

### Cell 15 — Chart 3: Grid Dependency

**What it shows:** Hour-by-hour grid interaction.

- **Orange bars (upward)** — Grid imports (drawing from grid, kW)
- **Blue bars (downward)** — Grid exports (sending to grid, kW)

A system achieving high self-sufficiency will have mostly zero import bars during daylight hours, with small imports only at night.

Saved as `chart3_grid_dependency_jaipur.png`.

---

### Cell 16 — Markdown: "6. Multi-City Simulation"

Section header. No code.

---

### Cell 17 — Multi-City Loop

**What it does:** Runs the full simulation for all 12 Indian cities, using 3 representative days per city (25th, 50th, and 75th percentile irradiance days) and the median load day for all. Results are collected into a list `city_results`.

**Why 3 days per city?**  
One day captures only a snapshot. Using the 25th percentile (cloudy day), median (average day), and 75th percentile (sunny day) shows the system's performance range — best case, typical, and below-average solar day.

**KPI averaging:**  
After running all 3 days, the KPIs are averaged. This `avg_kpi` per city is what gets used in all multi-city charts.

**Output structure per city:**
```python
{
    'city': 'jaipur',
    'climate_zone': 'BWh',
    'state': 'Rajasthan',
    'results': [result_day1, result_day2, result_day3],
    'kpis': [kpi_day1, kpi_day2, kpi_day3],
    'avg_kpi': {...averaged KPIs...}
}
```

---

### Cell 18 — Chart 4: Cross-City KPI Comparison

**What it shows:** A 4-panel comparison across all 12 cities.

- **Panel 1:** Self-sufficiency % (sorted descending, color-coded by climate zone)
- **Panel 2:** CO₂ avoided per day (kg)
- **Panel 3:** Solar self-consumption %
- **Panel 4:** Battery cycles per day

**Reading it:**  
Humid subtropical (Cwa) cities like Lucknow and Delhi often show surprisingly high self-sufficiency not because they have the most sun, but because their load/solar ratio is well-matched to the 5 kWh battery. Hot desert cities (BWh) like Jaipur and Jodhpur generate the most solar but sometimes export more, because the sun outpaces what the battery can store.

Saved as `chart4_cross_city_kpi_comparison.png`.

---

### Cell 19 — Chart 5: KPI Dashboard (Gauge Style)

**What it shows:** A radar/gauge-style dashboard for the demo city (Jaipur) showing 5 KPIs on circular gauges.

This is a visual summary intended for any "dashboard" or presentation context — shows self-sufficiency, solar self-consumption, CO₂ avoided, peak hours avoided, and battery utilization as circular fill gauges.

Saved as `chart5_kpi_dashboard_jaipur.png`.

---

### Cell 20 — Markdown: "7. Climate Zone Analysis"

Section header. No code.

---

### Cell 21 — Climate Zone Analysis + Chart 6

**What it does:** Groups the 12 cities by their Köppen climate zone and computes zone-level averages for all KPIs.

**The 5 climate zones in this study:**

| Code | Name | Indian Cities in Study |
|------|------|----------------------|
| BWh | Hot Desert | Jaipur, Jodhpur |
| BSh | Hot Semi-Arid | Ahmedabad, Hyderabad |
| Aw | Tropical Savanna | Chennai, Nagpur, Bangalore, Kolkata |
| Cwa | Humid Subtropical | Delhi, Guwahati, Lucknow |
| Am | Tropical Monsoon | Thiruvananthapuram |

**Chart 6:** A grouped bar chart comparing climate zones on self-sufficiency and CO₂ avoided, with color coding by zone.

**Key insight from this chart:**  
BSh and BWh zones (semi-arid and desert) have the highest solar potential but not always the highest self-sufficiency. This is because their irradiance is so high that the 5 kWh battery fills up by early afternoon, and all subsequent solar is exported rather than stored. A larger battery would dramatically improve these cities.

Saved as `chart6_climate_zone_analysis.png`.

---

### Cell 22 — Energy Balance Verification

**What it does:** A physics sanity check. Every simulation must obey conservation of energy.

**Three checks performed:**

```
1. LOAD BALANCE (every hour):
   solar_direct + battery_discharge + grid_import = load
   (all energy into the household must come from somewhere)

2. SOLAR BALANCE (every hour):
   solar_direct + battery_charge + grid_export = total_solar
   (all energy produced by the panel must go somewhere)

3. SOC BOUNDS (every hour):
   min_soc ≤ SOC ≤ max_soc
   (battery never exceeded its limits)
```

If any check fails, there's a bug in the dispatch logic. All three pass confirms the simulation is physically consistent.

**Why does this matter?**  
Energy simulations are easy to get wrong. A bug might silently create or destroy energy, making results look better than they are. This cell catches those errors.

---

### Cell 23 — Hourly Dispatch Table

**What it does:** Prints a formatted 24-row table showing every energy flow for the Jaipur demo day, hour by hour.

```
Hour  Solar(kW)  Load(kW)  Charge  Discharge  Import  Export  SOC(%)
00:00   0.000     0.412    0.000    0.250     0.162   0.000    17.2
01:00   0.000     0.387    0.000    0.250     0.137   0.000    14.3
...
12:00   2.092     0.734    1.358    0.000     0.000   0.000    86.4
...
```

This is the most detailed output in the notebook — useful for debugging or presenting granular dispatch decisions. It is also saved to `results/battery_optimizer/dispatch_schedule_jaipur.csv`.

---

### Cell 24 — Markdown: Results Summary

A markdown cell with a formatted results table. No code.

---

### Cell 25 — Final Summary

**What it does:** Prints the complete project summary including:
- Single-city KPIs for Jaipur
- Multi-city averages across all 12 cities
- Climate zone averages
- List of all 9 output files saved

**Example output:**
```
IEMS SOLAR BATTERY OPTIMIZER — FINAL SUMMARY
============================================================
Demo City (Jaipur, BWh):
  Self-Sufficiency:     87.8%
  CO₂ Avoided:         9.30 kg/day
  Peak Hours Avoided:  18/24
  Solar Self-Consump:  94.1%

Multi-City Average (12 cities):
  Self-Sufficiency:    71.2%
  CO₂ Avoided:        7.03 kg/day

Output files saved to: results/battery_optimizer/
  chart1_energy_flows_jaipur.png
  chart2_battery_soc_jaipur.png
  chart3_grid_dependency_jaipur.png
  chart4_cross_city_kpi_comparison.png
  chart5_kpi_dashboard_jaipur.png
  chart6_climate_zone_analysis.png
  load_profile_overview.png
  multi_city_battery_kpis.csv
  dispatch_schedule_jaipur.csv
```

---

## Key Terms Glossary

| Term | Definition |
|------|-----------|
| **GHI** | Global Horizontal Irradiance — the total solar energy hitting a flat horizontal surface, measured in W/m² (watts per square metre). The higher this is, the more solar the panels produce. |
| **STC** | Standard Test Conditions — the lab conditions used to rate solar panels (1000 W/m² irradiance, 25°C panel temperature). Real-world output is usually 10–25% lower. |
| **kW vs kWh** | kW is power (how fast energy flows — like speed). kWh is energy (how much energy over time — like distance). A 2.5 kW panel running for 2 hours produces 5 kWh. |
| **SOC** | State of Charge — the current energy level of the battery expressed as a percentage. 100% = fully charged, 0% = completely empty. |
| **C-rate** | How fast a battery charges or discharges relative to its capacity. 0.5C means it takes 2 hours to fully charge. Lower C-rates are gentler on the battery. |
| **Round-trip efficiency** | The ratio of energy you get out of a battery vs. energy you put in. 92% means for every 1 kWh stored, you get 0.92 kWh back. The other 8% became heat. |
| **Self-sufficiency** | The fraction of your electricity demand met without drawing from the grid. 90% self-sufficient = only 10% from grid. |
| **Solar self-consumption** | The fraction of your solar generation used at home (directly or via battery). 100% = no solar wasted. Low values mean you're exporting more than you're storing. |
| **Grid import** | Electricity drawn from the public power grid. |
| **Grid export** | Excess solar electricity pushed back to the public grid. In India, this earns money under net metering regulations. |
| **CO₂ emission factor** | How much carbon dioxide is emitted per kWh of grid electricity. India's is 0.82 kg/kWh because most power comes from coal. |
| **Look-ahead** | A decision-making strategy that looks at predicted future conditions before acting now. Here: checking the next 3 hours for a predicted energy deficit before deciding how much to charge the battery now. |
| **Dispatch** | Deciding when to charge, discharge, import, or export at each time step. "Dispatching" a battery means commanding it to act. |
| **Greedy algorithm** | An optimization approach that makes the locally best decision at each step without reconsidering prior decisions. The dispatch here is greedy — it doesn't solve a global optimization, just makes the best choice hour by hour. |
| **KPI** | Key Performance Indicator — a quantitative metric used to evaluate system performance. |
| **W/m²** | Watts per square metre — the unit of solar irradiance. On a clear summer day in India, GHI can reach 1000–1100 W/m² at noon. |
| **LoRA** | Low-Rank Adaptation — the fine-tuning technique used on Chronos. Instead of retraining all 200M parameters, only tiny "adapter" matrices are trained, making fine-tuning fast and storage-efficient. |
| **Chronos-T5** | Amazon's pre-trained time series foundation model. It was trained on millions of time series from many domains and can generalize to new forecasting tasks with minimal or no retraining. |
| **Köppen climate zone** | A climate classification system using letters (BWh, BSh, Aw, Cwa, Am) based on temperature and rainfall patterns. Used here to group Indian cities with similar solar irradiance behavior. |
| **ResIn / RevIN** | Reversible Instance Normalization — a technique applied during Chronos inference to make the model's inputs scale-invariant. The input is normalized, inference runs, then the output is denormalized back to real W/m² values. |
| **npy file** | NumPy binary file format. Stores arrays efficiently (faster and smaller than CSV). All predictions and window data in this project are stored as `.npy` files. |

---

## Data Flow Summary

```
results/pred_chronos_finetuned_{city}.npy    (shape: 729 × 24, W/m²)
        │
        │  [Cell 5: loaded for all 12 cities]
        │
        ▼
irradiance_24h  ── [Cell 12/17: pick 75th pct sunny day]
        │
        │  [Cell 10: irradiance_to_solar_kw()]
        │  solar = (GHI/1000) × 2.5 × 0.85 × 0.96
        ▼
solar_24h  (kW, 24 values)
        │
        │  [Cell 10: simulate_battery_dispatch()]
        │  hour-by-hour greedy dispatch
        ├──────────────────────────────────────────────────────┐
        │                                                       │
UCI household data ──[Cell 7: resample + ×0.5]──▶ load_24h   │
                                                    (kW, 24)  │
                                                               │
        ◀──────────────────────── result dict with 24h arrays ─┘
        │
        │  [Cell 10: compute_kpis()]
        ▼
KPI dict: 11 scalar metrics per day
        │
        │  [Cell 17: repeated for 12 cities × 3 days]
        ▼
df_kpis: 12 rows (one per city, averaged across 3 days)
        │
        ├──▶ Chart 4: Cross-city comparison
        ├──▶ Chart 5: Dashboard for Jaipur
        ├──▶ Chart 6: Climate zone aggregation
        └──▶ multi_city_battery_kpis.csv
```

---

## Why Each Design Choice Was Made

| Choice | Why |
|--------|-----|
| 2.5 kW panel | Typical affordable Indian rooftop size; larger systems require structural roof assessment |
| 5 kWh battery | Common residential product size in India (BYD, Luminous); affordable for middle class |
| 0.5C charge rate | Maximizes battery life; aggressive charging (1C+) degrades cells faster |
| 72h context for Chronos | 3 days captures multi-day weather cycles (cloud formation, monsoon patterns) |
| 24h forecast horizon | "Day-ahead" forecasting — actionable for next-day energy planning |
| Median load day (not mean) | Median is more representative; mean is pulled up by rare high-consumption outliers |
| 75th percentile sunny day | Shows good performance without cherry-picking the best possible day |
| 0.82 kg/kWh CO₂ | Official India CEA figure for 2023–24; using 0.40 (European average) would halve the apparent climate benefit |
| UCI dataset × 0.5 | No publicly available Indian minute-level smart meter dataset exists; scaling is the best approximation |
