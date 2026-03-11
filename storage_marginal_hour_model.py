#!/usr/bin/env python3
"""
RFC Power — Storage Duration Economics Model
=============================================
Marginal Hour Analysis: When does adding another hour stop paying?

Compares RFC, Li-Ion, and Form Energy across storage durations using:
  - Perfect foresight dispatch (rolling window LP-equivalent)
  - Annualised technology costs (power + energy components)
  - Marginal revenue vs marginal cost crossover analysis

Data: Accepts real hourly price data (CSV) or falls back to calibrated
      synthetic UK profiles.

Usage:
  python storage_marginal_hour_model.py                    # synthetic prices
  python storage_marginal_hour_model.py prices.csv         # real price CSV
  python storage_marginal_hour_model.py prices.csv GBP     # specify currency column

CSV format expected: Date/datetime column + price column (£/MWh or EUR/MWh).
If EUR, set EUR_TO_GBP conversion factor below.
"""

import numpy as np
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linprog
from scipy.sparse import csr_matrix

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — skipping charts, outputting data only.")

# =============================================================================
# CONFIGURATION — Edit these to match your data and assumptions
# =============================================================================

# Technology parameters (USD)
TECHNOLOGIES = {
    'RFC': {
        'power_usd_kw': 430,     # $/kW power component
        'energy_usd_kwh': 15,    # $/kWh energy (tank) component
        'rte': 100,             # Round-trip efficiency
        'color': '#2563eb',
    },
    'Li-Ion': {
        'power_usd_kw': 0,       # Flat cost — all in energy
        'energy_usd_kwh': 75,    # $/kWh (all-in)
        'rte': 75,
        'color': '#16a34a',
    },
    'Form Energy': {
        'power_usd_kw': 500,     # $/kW power component
        'energy_usd_kwh': 15,    # $/kWh energy component
        'rte': 0.40,
        'color': '#dc2626',
    },
}

# Financial assumptions
WACC = 0.08
ASSET_LIFE_YEARS = 20
USD_TO_GBP = 0.80
EUR_TO_GBP = 0.86  # Used if price data is in EUR

# Durations to evaluate (hours)
DURATIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 100]

# Dispatch method: True = LP optimal (slower, accurate); False = greedy heuristic (fast)
USE_LP_DISPATCH = True

# Dispatch optimisation (greedy fallback only)
DISPATCH_WINDOW_MULTIPLIER = 4  # Window = max(48, duration * this)
DISPATCH_MAX_WINDOW = 336       # Cap at 2 weeks
DISPATCH_STEP = 24              # Advance 24h per iteration


# =============================================================================
# PRICE PROFILES — Synthetic (calibrated to UK market stats)
# =============================================================================

def generate_uk_2024_prices(n_days=365, seed=42):
    """Synthetic UK 2024 day-ahead prices. Calibrated to ~£75/MWh mean, £45 daily spread."""
    np.random.seed(seed)
    hours = np.arange(n_days * 24)
    n = len(hours)
    hod = hours % 24
    doy = (hours // 24) % 365

    seasonal = 75 + 25 * np.cos(2 * np.pi * (doy - 15) / 365)
    daily_shape = np.array([-20,-20,-22,-22,-18,-15, 5,15,20,18,15,12,
                             8,5,8,15,25,30,28,20,10,0,-8,-15])
    prices = seasonal + daily_shape[hod]
    prices += np.repeat(np.random.normal(0, 15, n_days), 24)[:n]
    prices += np.random.normal(0, 8, n)
    prices += (np.random.random(n) < 0.02) * np.random.uniform(-40, -80, n)
    prices += (np.random.random(n) < 0.01) * np.random.uniform(80, 250, n)
    prices += ((hours // 24) % 7 >= 5).astype(float) * (-15)
    return prices


def generate_uk_2030_prices(n_days=365, seed=42):
    """Synthetic UK 2030 prices. Higher RE penetration, more volatile, negative pricing."""
    np.random.seed(seed + 1)
    hours = np.arange(n_days * 24)
    n = len(hours)
    hod = hours % 24
    doy = (hours // 24) % 365

    seasonal = 55 + 20 * np.cos(2 * np.pi * (doy - 15) / 365)
    daily_shape = np.array([-15,-18,-20,-20,-15,-10, 0,8,5,-10,-20,-25,
                            -28,-25,-20,-5,25,40,45,35,15,5,-5,-10])
    prices = seasonal + daily_shape[hod]

    # Solar midday dip (stronger in summer)
    summer = np.maximum(0, np.cos(2 * np.pi * (doy - 172) / 365))
    solar_dip = np.array([0,0,0,0,0,0, 0,0,0,-15,-20,-25,-25,-20,-15,0, 0,0,0,0,0,0,0,0])
    prices += solar_dip[hod] * np.repeat(summer, 24)[:n]

    prices += np.repeat(np.random.normal(0, 20, n_days), 24)[:n]
    prices += np.random.normal(0, 10, n)
    prices += (np.random.random(n) < 0.08) * np.random.uniform(-30, -100, n)
    prices += (np.random.random(n) < 0.008) * np.random.uniform(100, 350, n)
    prices += ((hours // 24) % 7 >= 5).astype(float) * (-12)
    return prices


def generate_uk_2040_prices(n_days=365, seed=42):
    """Synthetic UK 2040 prices. Deep RE penetration, frequent negatives, Dunkelflaute events."""
    np.random.seed(seed + 2)
    hours = np.arange(n_days * 24)
    n = len(hours)
    hod = hours % 24
    doy = (hours // 24) % 365

    seasonal = 40 + 15 * np.cos(2 * np.pi * (doy - 15) / 365)
    daily_shape = np.array([-15,-18,-20,-20,-15,-10, -5,0,-5,-25,-35,-40,
                            -40,-35,-25,-5,30,50,55,40,20,10,0,-10])
    prices = seasonal + daily_shape[hod]

    summer = np.maximum(0, np.cos(2 * np.pi * (doy - 172) / 365))
    solar_dip = np.array([0,0,0,0,0,0, 0,0,-5,-20,-30,-35,-35,-30,-20,-5, 0,0,0,0,0,0,0,0])
    prices += solar_dip[hod] * np.repeat(summer, 24)[:n]

    # Multi-day weather correlation (wind lulls)
    weather = np.zeros(n_days)
    weather[0] = np.random.normal(0, 1)
    for d in range(1, n_days):
        weather[d] = 0.85 * weather[d-1] + np.random.normal(0, 0.55)
    prices += np.repeat(weather * 30, 24)[:n]

    prices += np.random.normal(0, 12, n)
    prices += (np.random.random(n) < 0.15) * np.random.uniform(-20, -80, n)

    # Dunkelflaute scarcity events
    for _ in range(7):
        sd = np.random.randint(0, n_days - 5)
        dur = np.random.randint(2, 6)
        sh, eh = sd * 24, min((sd + dur) * 24, n)
        prices[sh:eh] += np.random.uniform(60, 180)

    prices += ((hours // 24) % 7 >= 5).astype(float) * (-10)
    return prices


# =============================================================================
# LOAD REAL PRICE DATA
# =============================================================================

def load_price_csv(filepath, price_col=None, date_col=None, currency='GBP',
                   recent_year_only=False):
    """
    Load hourly price data from CSV.

    Expects columns for date/time and price. Will auto-detect columns if not specified.
    Handles Ember format (Country, ISO3 Code, Date, Price (EUR/MWhe))
    and generic format (datetime, price).
    """
    import pandas as pd

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Columns: {list(df.columns)}")

    # Auto-detect price column
    if price_col is None:
        price_candidates = [c for c in df.columns if 'price' in c.lower() or 'eur' in c.lower() or 'gbp' in c.lower()]
        if price_candidates:
            price_col = price_candidates[0]
        else:
            # Assume last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            price_col = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[-1]
    print(f"Using price column: {price_col}")

    # Auto-detect date column
    if date_col is None:
        date_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'hour' in c.lower()]
        date_col = date_candidates[0] if date_candidates else df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if recent_year_only:
        year_counts    = df.groupby(df[date_col].dt.year).size()
        target_year    = int(year_counts[year_counts >= 8700].index.max())
        df             = df[df[date_col].dt.year == target_year].reset_index(drop=True)
        print(f"Filtered to most recent complete year: {target_year} ({len(df)} hours)")

    prices = df[price_col].values.astype(float)

    # Convert currency if needed
    if currency.upper() == 'EUR':
        prices = prices * EUR_TO_GBP
        print(f"Converted EUR -> GBP at {EUR_TO_GBP}")

    # Handle NaN
    nan_count = np.isnan(prices).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values — interpolating")
        prices = np.interp(
            np.arange(len(prices)),
            np.where(~np.isnan(prices))[0],
            prices[~np.isnan(prices)]
        )

    print(f"Price stats: mean={np.mean(prices):.1f}, std={np.std(prices):.1f}, "
          f"min={np.min(prices):.1f}, max={np.max(prices):.1f}")
    print(f"Hours: {len(prices)} ({len(prices)/24:.0f} days)")

    return prices


# =============================================================================
# DISPATCH OPTIMISATION — Perfect Foresight
# =============================================================================

def optimal_dispatch_annual(prices, duration, rte=0.75, power_mw=1.0):
    """
    Perfect foresight dispatch using rolling window sort-based optimisation.

    For each window:
    1. Sort prices to identify cheapest (charge) and most expensive (discharge) hours
    2. Pair them respecting capacity constraints
    3. Forward-simulate with SoC tracking

    Returns: total annual revenue (£/MW/year)
    """
    n = len(prices)
    capacity = power_mw * duration

    window = max(48, min(duration * DISPATCH_WINDOW_MULTIPLIER, DISPATCH_MAX_WINDOW))
    step = DISPATCH_STEP

    dispatch = np.zeros(n)
    soc = np.zeros(n + 1)

    for start in range(0, n - step, step):
        end = min(start + window, n)
        wp = prices[start:end]
        w_len = end - start

        sorted_idx = np.argsort(wp)

        max_cycles = max(1, w_len // (2 * max(duration, 1)))
        n_pairs = int(min(duration * max_cycles, w_len // 2))

        charge_hours = set()
        discharge_hours = set()

        for i in range(min(n_pairs, w_len)):
            low_idx = sorted_idx[i]
            high_idx = sorted_idx[-(i+1)]
            if low_idx == high_idx:
                continue
            if wp[high_idx] * rte > wp[low_idx]:
                charge_hours.add(int(low_idx))
                discharge_hours.add(int(high_idx))
            else:
                break

        # Remove any hour assigned to both
        overlap = charge_hours & discharge_hours
        charge_hours -= overlap
        discharge_hours -= overlap

        # Forward simulate for step hours only
        for h in range(min(step, w_len)):
            abs_h = start + h
            if abs_h >= n:
                break

            if h in charge_hours and soc[abs_h] < capacity - 0.01:
                charge = min(power_mw, capacity - soc[abs_h])
                dispatch[abs_h] = -charge
                soc[abs_h + 1] = soc[abs_h] + charge
            elif h in discharge_hours and soc[abs_h] > 0.01:
                disc = min(power_mw, soc[abs_h])
                dispatch[abs_h] = disc
                soc[abs_h + 1] = soc[abs_h] - disc
            else:
                soc[abs_h + 1] = soc[abs_h]

    # Revenue calculation
    revenue = 0.0
    for h in range(n):
        if dispatch[h] > 0:    # Discharging — earn revenue (net of RTE)
            revenue += dispatch[h] * prices[h] * rte
        elif dispatch[h] < 0:  # Charging — pay for energy
            revenue += dispatch[h] * prices[h]

    return revenue


def lp_dispatch_annual(prices, duration, rte=0.75, power_mw=1.0):
    """
    Provably-optimal perfect-foresight dispatch via Linear Programme.

    Variables per hour t:  charge[t], discharge[t] in [0, power_mw]
                           soc[t+1]  in [0, capacity]   (soc[0] = 0 fixed)
    Equality:              soc[t+1] = soc[t] + charge[t] - discharge[t]
    Objective:             maximise sum(discharge*price*rte) - sum(charge*price)

    Returns: total annual revenue (GBP/MW/year)
    """
    n        = len(prices)
    capacity = power_mw * duration
    nv       = 3 * n   # charge(n) + discharge(n) + soc[1..n](n)

    # Objective: minimise -(revenue) = sum(charge*price) - sum(discharge*price*rte)
    c_obj = np.concatenate([prices, -prices * rte, np.zeros(n)])

    # Sparse equality matrix: soc[t+1] = soc[t] + charge[t] - discharge[t]
    t_idx = np.arange(n)
    rows  = np.concatenate([t_idx, t_idx, t_idx, t_idx[1:]])
    cols  = np.concatenate([t_idx, n + t_idx, 2*n + t_idx, 2*n + t_idx[1:] - 1])
    vals  = np.concatenate([-np.ones(n), np.ones(n), np.ones(n), -np.ones(n - 1)])
    A_eq  = csr_matrix((vals, (rows, cols)), shape=(n, nv))
    b_eq  = np.zeros(n)

    bounds = [(0, power_mw)] * n + [(0, power_mw)] * n + [(0, capacity)] * n

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                     method='highs', options={'disp': False})
    if result.status != 0:
        raise RuntimeError(f"LP solver failed (status {result.status}): {result.message}")

    charge    = result.x[:n]
    discharge = result.x[n:2*n]
    return float(np.dot(discharge, prices) * rte - np.dot(charge, prices))


# =============================================================================
# COST MODEL — Annualised Capex
# =============================================================================

def annuity_factor(wacc=WACC, years=ASSET_LIFE_YEARS):
    return wacc / (1 - (1 + wacc)**(-years))


def total_annual_cost(duration_h, power_usd_kw, energy_usd_kwh):
    """Annualised total cost for a system of given duration (£/MW/year)."""
    capex_usd_per_mw = power_usd_kw * 1000 + energy_usd_kwh * 1000 * duration_h
    capex_gbp = capex_usd_per_mw * USD_TO_GBP
    return capex_gbp * annuity_factor()


def marginal_cost_nth_hour(n, power_usd_kw, energy_usd_kwh):
    """
    Annualised cost of adding the nth hour of storage (£/MW/year).
    Hour 1 bears the full power cost + 1h energy.
    Hours 2+ only cost the incremental energy component.
    """
    if n <= 1:
        capex = (power_usd_kw * 1000 + energy_usd_kwh * 1000) * USD_TO_GBP
    else:
        capex = (energy_usd_kwh * 1000) * USD_TO_GBP
    return capex * annuity_factor()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_marginal_analysis(prices, label='UK 2024', durations=None):
    """
    Run the full marginal hour analysis for all technologies.

    Returns dict with:
      - durations
      - For each tech: total_revenue[], marginal_revenue[], marginal_cost[],
        total_cost[], net_value[], crossover_hour, peak_net_duration
    """
    if durations is None:
        durations = DURATIONS

    af = annuity_factor()
    results = {}

    for tech_name, params in TECHNOLOGIES.items():
        rte = params['rte']
        power = params['power_usd_kw']
        energy = params['energy_usd_kwh']

        dispatch_fn = lp_dispatch_annual if USE_LP_DISPATCH else optimal_dispatch_annual
        method_tag  = 'LP' if USE_LP_DISPATCH else 'greedy'
        print(f"\n  {tech_name} ({rte*100:.0f}% RTE, {method_tag}):", end='', flush=True)

        # Revenue at each duration
        revenues = []
        for d in durations:
            rev = dispatch_fn(prices, d, rte=rte)
            revenues.append(rev)
            if d in [1, 4, 8, 12, 24, 48, 100]:
                print(f" {d}h=£{rev/1000:.0f}k", end='', flush=True)
        print()

        # Marginal revenue
        marginal_rev = []
        for i in range(len(durations)):
            if i == 0:
                marginal_rev.append(revenues[0])
            else:
                delta = revenues[i] - revenues[i-1]
                step = durations[i] - durations[i-1]
                marginal_rev.append(delta / step)

        # Marginal cost (energy-only for hours > 1)
        mc_energy = energy * 1000 * USD_TO_GBP * af
        mc_first = (power * 1000 + energy * 1000) * USD_TO_GBP * af
        marginal_cost = [mc_first if d == 1 else mc_energy for d in durations]

        # Total cost and net value
        total_costs = [total_annual_cost(d, power, energy) for d in durations]
        net_values = [r - c for r, c in zip(revenues, total_costs)]

        # Crossover: where marginal revenue < marginal cost (hours > 1)
        crossover = None
        for i in range(1, len(durations)):
            if marginal_rev[i] < marginal_cost[i]:
                if i > 1:
                    mr_prev, mc_prev = marginal_rev[i-1], marginal_cost[i-1]
                    mr_curr, mc_curr = marginal_rev[i], marginal_cost[i]
                    denom = (mr_prev - mc_prev) - (mr_curr - mc_curr)
                    if abs(denom) > 0.01:
                        frac = (mr_prev - mc_prev) / denom
                        crossover = durations[i-1] + frac * (durations[i] - durations[i-1])
                    else:
                        crossover = float(durations[i])
                else:
                    crossover = float(durations[i])
                break

        # Peak net value
        peak_idx = int(np.argmax(net_values))

        results[tech_name] = {
            'total_revenue': revenues,
            'marginal_revenue': marginal_rev,
            'marginal_cost': marginal_cost,
            'total_cost': total_costs,
            'net_value': net_values,
            'crossover_hour': crossover,
            'peak_net_duration': durations[peak_idx],
            'peak_net_value': net_values[peak_idx],
            'energy_marginal_cost': mc_energy,
        }

    return {
        'durations': durations,
        'technologies': results,
        'price_label': label,
        'price_stats': {
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'pct_negative': float(np.mean(prices < 0) * 100),
            'hours': len(prices),
        },
        'assumptions': {
            'wacc': WACC,
            'asset_life': ASSET_LIFE_YEARS,
            'usd_gbp': USD_TO_GBP,
        },
    }


# =============================================================================
# CHARTING
# =============================================================================

def plot_marginal_analysis(analysis, output_path='marginal_hour_analysis.png',
                           max_duration=None):
    """Generate the 4-panel marginal hour analysis chart.

    max_duration: if set, plot only durations <= this value.
    """
    if not HAS_MPL:
        print("Skipping charts (matplotlib not available)")
        return

    durations = analysis['durations']
    techs     = analysis['technologies']

    # Optional subsetting — build filtered copies without mutating the analysis dict
    if max_duration is not None:
        keep      = [i for i, d in enumerate(durations) if d <= max_duration]
        durations = [durations[i] for i in keep]
        filtered  = {}
        for tech_name, t in techs.items():
            ft = dict(t)
            for key in ('total_revenue', 'marginal_revenue',
                        'marginal_cost', 'total_cost', 'net_value'):
                ft[key] = [t[key][i] for i in keep]
            # Recalculate peak within the visible range
            peak_idx              = int(np.argmax(ft['net_value']))
            ft['peak_net_duration'] = durations[peak_idx]
            ft['peak_net_value']    = ft['net_value'][peak_idx]
            filtered[tech_name]   = ft
        techs = filtered

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ---- PANEL 1: Marginal Revenue vs Marginal Cost (top, full width) ----
    ax1 = fig.add_subplot(gs[0, :])

    for tech_name, t in techs.items():
        color = TECHNOLOGIES[tech_name]['color']
        mr = [m / 1000 for m in t['marginal_revenue']]
        mc_energy = t['energy_marginal_cost'] / 1000

        ax1.plot(durations, mr, color=color, linewidth=2.5,
                 label=f'{tech_name} marginal revenue', zorder=3)
        ax1.axhline(y=mc_energy, color=color, linewidth=1.5, linestyle='--', alpha=0.5,
                     label=f'{tech_name} marginal cost (£{mc_energy:.1f}k)')

        cd       = t['crossover_hour']
        plot_max = max(durations)
        if cd and cd <= plot_max:
            ax1.plot(cd, mc_energy, 'o', color=color, markersize=10, zorder=5,
                     markeredgecolor='white', markeredgewidth=2)
            ax1.annotate(f'{cd:.0f}h', xy=(cd, mc_energy),
                         xytext=(min(cd + 2, plot_max * 0.92),
                                 mc_energy + max(mr) * 0.06),
                         fontsize=10, fontweight='bold', color=color,
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax1.axvspan(8, 24, alpha=0.05, color='#2563eb', zorder=0)
    ax1.text(16, max([m/1000 for m in techs['RFC']['marginal_revenue']]) * 0.85,
             'RFC\nsweet spot', ha='center', fontsize=9, color='#2563eb', alpha=0.5, style='italic')

    ax1.set_xlabel('Storage Duration (hours)', fontsize=12)
    ax1.set_ylabel('Marginal Value of Additional Hour (£k/MW/year)', fontsize=12)
    ax1.set_title('When Does Adding Another Hour Stop Paying?\n'
                   'Marginal Revenue of the nth Hour vs Marginal Cost of Tank Capacity',
                   fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2, loc='upper right')
    ax1.set_xlim(1, max(durations))
    ax1.set_ylim(bottom=-0.5)
    ax1.grid(alpha=0.15)

    # ---- PANEL 2: Net Annual Value ----
    ax2 = fig.add_subplot(gs[1, 0])
    for tech_name, t in techs.items():
        color = TECHNOLOGIES[tech_name]['color']
        net = [v / 1000 for v in t['net_value']]
        ax2.plot(durations, net, color=color, linewidth=2.5, label=tech_name)

        pd_dur = t['peak_net_duration']
        pd_val = t['peak_net_value'] / 1000
        if pd_dur <= max(durations):
            ax2.scatter([pd_dur], [pd_val], color=color, s=100, zorder=5,
                        edgecolors='white', linewidths=2)
            ax2.annotate(f'{pd_dur}h: £{pd_val:.0f}k', xy=(pd_dur, pd_val),
                         xytext=(min(pd_dur + 2, max(durations) * 0.85), pd_val + 2),
                         fontsize=9, fontweight='bold', color=color,
                         arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))

    ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Duration (hours)', fontsize=11)
    ax2.set_ylabel('Net Annual Value (£k/MW/year)', fontsize=11)
    ax2.set_title('Net Value = Revenue − Annualised Capex', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim(1, max(durations))
    ax2.grid(alpha=0.15)

    # ---- PANEL 3: Total Revenue ----
    ax3 = fig.add_subplot(gs[1, 1])
    for tech_name, t in techs.items():
        color = TECHNOLOGIES[tech_name]['color']
        rev = [r / 1000 for r in t['total_revenue']]
        cost = [c / 1000 for c in t['total_cost']]
        ax3.plot(durations, rev, color=color, linewidth=2.5, label=f'{tech_name} revenue')
        ax3.plot(durations, cost, color=color, linewidth=1.5, linestyle=':', alpha=0.6,
                 label=f'{tech_name} cost')

    ax3.set_xlabel('Duration (hours)', fontsize=11)
    ax3.set_ylabel('£k/MW/year', fontsize=11)
    ax3.set_title('Total Revenue vs Total Cost', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.set_xlim(1, max(durations))
    ax3.grid(alpha=0.15)

    # Source note
    stats = analysis['price_stats']
    fig.text(0.5, 0.005,
             f'{analysis["price_label"]} prices (mean £{stats["mean"]:.0f}/MWh, '
             f'{stats["pct_negative"]:.1f}% negative, {stats["hours"]} hours) | '
             f'Costs in USD converted at {USD_TO_GBP} USD/GBP | '
             f'{WACC*100:.0f}% WACC, {ASSET_LIFE_YEARS}yr life | '
             f'Perfect foresight dispatch | Excludes capacity market, ancillary services, O&M',
             ha='center', fontsize=7.5, color='grey', style='italic')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nChart saved: {output_path}")


def print_summary_table(analysis):
    """Print formatted summary to console."""
    durations = analysis['durations']
    techs = analysis['technologies']

    print(f"\n{'='*90}")
    print(f"MARGINAL HOUR ANALYSIS — {analysis['price_label']}")
    print(f"{'='*90}")

    stats = analysis['price_stats']
    print(f"Prices: mean £{stats['mean']:.0f}/MWh | std £{stats['std']:.0f} | "
          f"neg {stats['pct_negative']:.1f}% | {stats['hours']} hours")

    # Summary row per technology
    print(f"\n{'Technology':<16} {'RTE':>5} {'Crossover':>10} {'Peak Dur':>10} {'Peak Net £k':>12}")
    print("-" * 60)
    for tech_name, t in techs.items():
        rte = TECHNOLOGIES[tech_name]['rte']
        cd = t['crossover_hour']
        cd_str = f"{cd:.0f}h" if cd else ">100h"
        pd = t['peak_net_duration']
        pv = t['peak_net_value'] / 1000
        print(f"{tech_name:<16} {rte*100:>4.0f}% {cd_str:>10} {pd:>9}h {pv:>11.1f}")

    # Detailed table — show all evaluated durations
    key_durs = durations
    print(f"\n{'Dur':>5}", end='')
    for tech_name in techs:
        short = tech_name[:8]
        print(f" | {short+' Rev':>10} {short+' Cost':>10} {short+' Net':>10}", end='')
    print()
    print("-" * (5 + len(techs) * 35))

    for d in key_durs:
        if d not in durations:
            continue
        i = durations.index(d)
        print(f"{d:>4}h", end='')
        for tech_name, t in techs.items():
            rev = t['total_revenue'][i] / 1000
            cost = t['total_cost'][i] / 1000
            net = t['net_value'][i] / 1000
            print(f" | {rev:>9.1f}k {cost:>9.1f}k {net:>9.1f}k", end='')
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 60)
    print("RFC POWER — Storage Duration Economics Model")
    print("Marginal Hour Analysis")
    print("=" * 60)

    # Load prices
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        currency = sys.argv[2] if len(sys.argv) > 2 else 'GBP'
        print(f"\nLoading real price data: {filepath}")
        prices = load_price_csv(filepath, currency=currency, recent_year_only=True)
        label = f"UK 2025 real prices (LP dispatch)"
    else:
        print("\nNo price file provided — using synthetic UK 2024 profiles.")
        print("Usage: python storage_marginal_hour_model.py <prices.csv> [GBP|EUR]")
        prices = generate_uk_2024_prices()
        label = "UK 2024 (synthetic)"

    print(f"\nRunning marginal analysis across {len(DURATIONS)} durations...")
    analysis = run_marginal_analysis(prices, label=label)

    # Output
    print_summary_table(analysis)

    # Save JSON
    json_path = 'marginal_analysis_output.json'
    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=convert)
    print(f"\nData saved: {json_path}")

    # Charts — full range and zoomed to 24h
    plot_marginal_analysis(analysis)
    plot_marginal_analysis(analysis, output_path='marginal_hour_analysis_24h.png',
                           max_duration=24)

    # Also run 2030 and 2040 if using synthetic
    if len(sys.argv) <= 1:
        for gen_fn, lbl in [(generate_uk_2030_prices, 'UK 2030'),
                            (generate_uk_2040_prices, 'UK 2040')]:
            print(f"\n{'='*60}")
            print(f"Running {lbl}...")
            p = gen_fn()
            a = run_marginal_analysis(p, label=f"{lbl} (synthetic)")
            print_summary_table(a)
            safe_label = lbl.replace(' ', '_').lower()
            plot_marginal_analysis(a, output_path=f'marginal_hour_{safe_label}.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
