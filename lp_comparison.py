"""
LP vs Greedy Dispatch Comparison
==================================
Implements a true LP optimal dispatch alongside the existing greedy heuristic,
runs both on UK 2025 prices, and produces an interactive Plotly comparison chart.

The LP is the provably-optimal perfect-foresight solution for the given price
series. The gap between LP and greedy revenue shows how much the heuristic
leaves on the table.

LP formulation
--------------
Variables (per hour t = 0..n-1):
  charge[t]    in [0, power_mw]
  discharge[t] in [0, power_mw]
  soc[t+1]     in [0, capacity]         (soc[0] = 0 fixed)

Equality constraints (SoC transition):
  soc[t+1] = soc[t] + charge[t] - discharge[t]

Objective (maximise revenue):
  max  sum_t( discharge[t] * price[t] * rte  -  charge[t] * price[t] )

RTE is applied at discharge (consistent with the greedy model).
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── constants (mirror storage_marginal_hour_model.py) ────────────────────────
EUR_TO_GBP = 0.86
DISPATCH_WINDOW_MULTIPLIER = 4
DISPATCH_MAX_WINDOW = 336
DISPATCH_STEP = 24
RFC_RTE = 0.75


# =============================================================================
# LP DISPATCH
# =============================================================================

def lp_dispatch(prices, duration, rte=RFC_RTE, power_mw=1.0):
    """
    Provably-optimal perfect-foresight dispatch via Linear Programme.

    Variables: charge[t], discharge[t], soc[t+1] for t = 0..n-1
    Equality:  soc[t+1] = soc[t] + charge[t] - discharge[t], soc[0] = 0
    Bounds:    charge, discharge in [0, power_mw];  soc in [0, capacity]
    Objective: maximise sum(discharge * price * rte) - sum(charge * price)
    """
    n = len(prices)
    capacity = power_mw * duration
    nv = 3 * n  # charge(n) + discharge(n) + soc[1..n](n)

    # ── objective (linprog minimises, so negate revenue) ─────────────────────
    c_obj = np.concatenate([
        prices,           # cost of charging (positive → we want to minimise)
        -prices * rte,    # negative revenue from discharging
        np.zeros(n),      # soc variables: no direct cost
    ])

    # ── equality constraints: soc transition ─────────────────────────────────
    # Row t: -charge[t] + discharge[t] + soc[t+1] - soc[t] = 0
    # Variable index mapping:
    #   charge[t]   → col t
    #   discharge[t]→ col n+t
    #   soc[t+1]    → col 2n+t
    #   soc[t]      → col 2n+t-1  (t >= 1;  soc[0]=0 so no variable for t=0)

    t_idx = np.arange(n)

    rows = np.concatenate([t_idx, t_idx, t_idx, t_idx[1:]])
    cols = np.concatenate([t_idx,          # charge[t]
                           n + t_idx,       # discharge[t]
                           2*n + t_idx,     # soc[t+1]
                           2*n + t_idx[1:] - 1])  # soc[t] for t>=1
    vals = np.concatenate([-np.ones(n),    # charge: -1
                            np.ones(n),    # discharge: +1
                            np.ones(n),    # soc[t+1]: +1
                            -np.ones(n-1)])  # soc[t]: -1

    A_eq = csr_matrix((vals, (rows, cols)), shape=(n, nv))
    b_eq = np.zeros(n)

    # ── variable bounds ───────────────────────────────────────────────────────
    bounds = ([(0, power_mw)] * n      # charge
            + [(0, power_mw)] * n      # discharge
            + [(0, capacity)] * n)     # soc[1..n]

    # ── solve ─────────────────────────────────────────────────────────────────
    result = linprog(
        c_obj,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={'disp': False},
    )

    if result.status != 0:
        raise RuntimeError(f"LP solver failed (status {result.status}): {result.message}")

    charge    = result.x[:n]
    discharge = result.x[n:2*n]
    soc       = np.concatenate([[0.0], result.x[2*n:]])   # prepend soc[0]=0

    revenue = float(np.dot(discharge, prices) * rte - np.dot(charge, prices))
    return charge, discharge, soc[:n], revenue


# =============================================================================
# GREEDY DISPATCH (copied from storage_marginal_hour_model.py for self-containment)
# =============================================================================

def greedy_dispatch(prices, duration, rte=RFC_RTE, power_mw=1.0):
    """Rolling-window greedy heuristic (identical to the main model)."""
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
            low_idx  = sorted_idx[i]
            high_idx = sorted_idx[-(i + 1)]
            if low_idx == high_idx:
                continue
            if wp[high_idx] * rte > wp[low_idx]:
                charge_hours.add(int(low_idx))
                discharge_hours.add(int(high_idx))
            else:
                break

        overlap = charge_hours & discharge_hours
        charge_hours    -= overlap
        discharge_hours -= overlap

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

    charge_arr    = np.where(dispatch < 0, -dispatch, 0.0)
    discharge_arr = np.where(dispatch > 0,  dispatch, 0.0)
    revenue = float(np.dot(discharge_arr, prices) * rte - np.dot(charge_arr, prices))
    return charge_arr, discharge_arr, soc[:n], revenue


# =============================================================================
# DATA LOADING
# =============================================================================

def load_most_recent_year(filepath):
    df = pd.read_csv(filepath)
    date_col  = 'Datetime (UTC)'
    price_col = 'Price (EUR/MWhe)'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    year_counts    = df.groupby(df[date_col].dt.year).size()
    complete_years = year_counts[year_counts >= 8700].index
    target_year    = int(complete_years.max())

    df_year = df[df[date_col].dt.year == target_year].copy().reset_index(drop=True)
    prices  = df_year[price_col].values.astype(float) * EUR_TO_GBP

    nan_mask = np.isnan(prices)
    if nan_mask.any():
        prices = np.interp(np.arange(len(prices)),
                           np.where(~nan_mask)[0], prices[~nan_mask])

    datetimes = df_year[date_col].values
    print(f"Loaded {target_year}: {len(prices)} hours  "
          f"mean GBP{prices.mean():.1f}/MWh  std {prices.std():.1f}  "
          f"neg {(prices < 0).mean()*100:.1f}%")
    return datetimes, prices, target_year


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(results):
    print(f"\n{'='*72}")
    print(f"{'DISPATCH COMPARISON':^72}")
    print(f"{'='*72}")
    print(f"{'Duration':<10} {'Method':<10} {'Revenue £/MW/yr':>16} {'Discharge h':>12} "
          f"{'LP efficiency':>14}")
    print("-" * 72)
    for d, r in results.items():
        lp_rev   = r['lp']['revenue']
        gr_rev   = r['greedy']['revenue']
        lp_dh    = r['lp']['discharge_hours']
        gr_dh    = r['greedy']['discharge_hours']
        eff      = gr_rev / lp_rev * 100 if lp_rev > 0 else float('nan')
        print(f"{d}h{'':<8} {'LP':<10} {lp_rev:>15,.0f}  {lp_dh:>11}  {'(benchmark)':>14}")
        print(f"{'':<10} {'Greedy':<10} {gr_rev:>15,.0f}  {gr_dh:>11}  {eff:>13.1f}%")
        print(f"{'':<10} {'Gap':<10} {(lp_rev-gr_rev):>15,.0f}  "
              f"{'':<11}  {(lp_rev-gr_rev)/1000:>10.1f}k left")
        print()


# =============================================================================
# INTERACTIVE CHART
# =============================================================================

def build_chart(datetimes, prices, year, results, output='lp_comparison.html'):
    dt_index = pd.to_datetime(datetimes)

    colours = {4: '#2563eb', 12: '#16a34a'}
    fills   = {4: 'rgba(37,99,235,0.10)', 12: 'rgba(22,163,74,0.10)'}

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.40, 0.35],
        vertical_spacing=0.06,
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=[
            f'UK Wholesale Price {year} (GBP/MWh)',
            f'State of Charge — RFC ({RFC_RTE*100:.0f}% RTE)   [solid = LP,  dashed = Greedy]',
            'Cumulative Revenue (GBP/kW)   [solid = LP,  dashed = Greedy]'
            '          |          Discharge hours (right axis)',
        ]
    )

    # ── Row 1: price ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scattergl(x=dt_index, y=prices, mode='lines',
                     name='Price (GBP/MWh)',
                     line=dict(color='#475569', width=0.8),
                     hovertemplate='%{x|%d %b %H:%M}<br>GBP%{y:.1f}/MWh<extra></extra>'),
        row=1, col=1
    )
    fig.add_hline(y=0, line=dict(color='#94a3b8', width=1, dash='dot'), row=1, col=1)

    # ── Row 2: SoC — LP (solid) vs Greedy (dashed) ───────────────────────────
    max_dur = max(results.keys())

    for d, r in results.items():
        colour = colours[d]
        fill   = fills[d]

        fig.add_trace(
            go.Scatter(x=dt_index, y=r['lp']['soc'],
                       mode='lines', name=f'{d}h LP SoC',
                       line=dict(color=colour, width=1.8),
                       fill='tozeroy', fillcolor=fill,
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h LP SoC: %{{y:.2f}} h<extra></extra>'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dt_index, y=r['greedy']['soc'],
                       mode='lines', name=f'{d}h Greedy SoC',
                       line=dict(color=colour, width=1.4, dash='dash'),
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h Greedy SoC: %{{y:.2f}} h<extra></extra>'),
            row=2, col=1
        )
        fig.add_hline(y=d, line=dict(color=colour, width=1, dash='dot'),
                      annotation_text=f'{d}h max',
                      annotation_font=dict(color=colour, size=9),
                      row=2, col=1)

    # ── Row 3: Cumulative revenue (left) + discharge hours (right) ────────────
    for d, r in results.items():
        colour = colours[d]

        lp_rev_kw     = r['lp']['cum_revenue_kw']
        greedy_rev_kw = r['greedy']['cum_revenue_kw']
        lp_dh_cum     = r['lp']['cum_discharge_h']
        greedy_dh_cum = r['greedy']['cum_discharge_h']
        lp_total      = r['lp']['revenue'] / 1000
        gr_total      = r['greedy']['revenue'] / 1000

        fig.add_trace(
            go.Scatter(x=dt_index, y=lp_rev_kw,
                       mode='lines', name=f'{d}h LP rev (GBP{lp_total:.0f}/kW)',
                       line=dict(color=colour, width=1.8),
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h LP: GBP%{{y:.1f}}/kW<extra></extra>'),
            row=3, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=dt_index, y=greedy_rev_kw,
                       mode='lines', name=f'{d}h Greedy rev (GBP{gr_total:.0f}/kW)',
                       line=dict(color=colour, width=1.4, dash='dash'),
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h Greedy: GBP%{{y:.1f}}/kW<extra></extra>'),
            row=3, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=dt_index, y=lp_dh_cum,
                       mode='lines', name=f'{d}h LP discharge h',
                       line=dict(color=colour, width=1.0, dash='dot'),
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h LP disch: %{{y:.0f}} h<extra></extra>'),
            row=3, col=1, secondary_y=True
        )
        fig.add_trace(
            go.Scatter(x=dt_index, y=greedy_dh_cum,
                       mode='lines', name=f'{d}h Greedy discharge h',
                       line=dict(color=colour, width=0.9, dash='dashdot'),
                       hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h Greedy disch: %{{y:.0f}} h<extra></extra>'),
            row=3, col=1, secondary_y=True
        )

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f'LP Optimal vs Greedy Dispatch — UK {year} Real Prices  '
                  f'(RFC, {RFC_RTE*100:.0f}% RTE)'),
            font=dict(size=15),
        ),
        height=1020,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                    font=dict(size=9)),
        margin=dict(l=70, r=80, t=90, b=60),
        template='plotly_white',
    )

    fig.update_yaxes(title_text='GBP/MWh', row=1, col=1)
    fig.update_yaxes(title_text='SoC (hours, 1 MW)',
                     range=[-0.1, max_dur + 0.5], row=2, col=1)
    fig.update_yaxes(title_text='Cumulative revenue (GBP/kW)',
                     secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text='Cumulative discharge (hours)',
                     secondary_y=True, row=3, col=1)

    fig.update_xaxes(
        title_text='Date',
        rangeslider=dict(visible=True, thickness=0.03),
        row=3, col=1
    )
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=7,  label='1W',  step='day',   stepmode='backward'),
            dict(count=1,  label='1M',  step='month', stepmode='backward'),
            dict(count=3,  label='3M',  step='month', stepmode='backward'),
            dict(count=6,  label='6M',  step='month', stepmode='backward'),
            dict(step='all', label='All'),
        ]),
        row=1, col=1
    )

    fig.write_html(output, include_plotlyjs='cdn')
    print(f"\nChart saved: {output}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    filepath = (sys.argv[1] if len(sys.argv) > 1
                else 'european_wholesale_electricity_price_data_hourly/United Kingdom.csv')
    durations = (4, 12)

    print(f"\nLoading: {filepath}")
    datetimes, prices, year = load_most_recent_year(filepath)

    results = {}
    for d in durations:
        results[d] = {}
        print(f"\n--- {d}h storage ---")

        # LP
        t0 = time.time()
        print(f"  LP solve...", end=' ', flush=True)
        c, dc, soc_lp, rev_lp = lp_dispatch(prices, duration=d)
        lp_s = time.time() - t0
        print(f"done in {lp_s:.1f}s  revenue GBP{rev_lp/1000:.1f}k/MW")
        results[d]['lp'] = {
            'charge': c, 'discharge': dc, 'soc': soc_lp,
            'revenue': rev_lp,
            'discharge_hours': int(np.sum(dc > 1e-4)),
            'cum_revenue_kw': np.cumsum(dc * prices * RFC_RTE - c * prices) / 1000,
            'cum_discharge_h': np.cumsum(dc > 1e-4).astype(float),
        }

        # Greedy
        t0 = time.time()
        print(f"  Greedy...", end=' ', flush=True)
        c, dc, soc_g, rev_g = greedy_dispatch(prices, duration=d)
        gr_s = time.time() - t0
        print(f"done in {gr_s:.1f}s  revenue GBP{rev_g/1000:.1f}k/MW")
        results[d]['greedy'] = {
            'charge': c, 'discharge': dc, 'soc': soc_g,
            'revenue': rev_g,
            'discharge_hours': int(np.sum(dc > 1e-4)),
            'cum_revenue_kw': np.cumsum(dc * prices * RFC_RTE - c * prices) / 1000,
            'cum_discharge_h': np.cumsum(dc > 1e-4).astype(float),
        }

    print_comparison(results)
    build_chart(datetimes, prices, year, results)
