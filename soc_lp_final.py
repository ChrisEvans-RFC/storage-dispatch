"""
SOC Final Visualisation — LP Optimal Dispatch
==============================================
Loads European day-ahead price data for a configurable year range,
runs LP-optimal perfect-foresight dispatch for configurable storage durations,
and produces a clean 2-panel interactive Plotly chart.

Panel 1: Hourly price with charge/discharge markers
Panel 2: State of Charge in hours (1 MW system), with capacity ceiling lines
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EUR_TO_GBP = 0.86

# Default technology assignments used by the CLI entry point
DURATION_TECH = {
    4:   {'name': 'LFP',  'rte': 0.85},
    12:  {'name': 'RFC',  'rte': 0.75},
    100: {'name': 'LDES', 'rte': 0.40},
}

# Sub-hourly SOC display resolution (hours); 0.5 = 30-min interpolation
SOC_DISPLAY_STEP = 0.5

# Colour palette for up to 4 technologies (assigned by position, not duration)
PALETTE = [
    ('#2563eb', 'rgba(37,99,235,0.12)'),
    ('#16a34a', 'rgba(22,163,74,0.12)'),
    ('#dc2626', 'rgba(220,38,38,0.12)'),
    ('#7c3aed', 'rgba(124,58,237,0.12)'),
]


# =============================================================================
# LP DISPATCH
# =============================================================================

def lp_dispatch(prices, duration, rte=0.75, power_mw=1.0):
    """
    Optimal perfect-foresight dispatch via Linear Programme (HiGHS backend).

    SoC is measured in available discharge hours (at power_mw).
    RTE losses occur on the charging side: 1 MWh charged → rte MWh stored.
    Charging and discharging power are both capped at power_mw.

    Variables per hour t:  charge[t], discharge[t] in [0, power_mw]
                           soc[t+1] in [0, capacity]   (soc[0]=0 fixed)
    Equality:              soc[t+1] = soc[t] + charge[t]*rte - discharge[t]
    Objective:             maximise sum(discharge*price - charge*price)
    """
    n        = len(prices)
    capacity = power_mw * duration
    nv       = 3 * n   # charge(n) + discharge(n) + soc[1..n](n)

    c_obj = np.concatenate([prices, -prices, np.zeros(n)])

    # Sparse equality matrix: soc transition
    # soc[t+1] = soc[t] + charge[t]*rte - discharge[t]
    t = np.arange(n)
    rows = np.concatenate([t, t, t, t[1:]])
    cols = np.concatenate([t, n + t, 2*n + t, 2*n + t[1:] - 1])
    vals = np.concatenate([-rte * np.ones(n), np.ones(n), np.ones(n), -np.ones(n - 1)])
    A_eq = csr_matrix((vals, (rows, cols)), shape=(n, nv))
    b_eq = np.zeros(n)

    bounds = [(0, power_mw)] * n + [(0, power_mw)] * n + [(0, capacity)] * n

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs', options={'disp': False})
    if res.status != 0:
        raise RuntimeError(f"LP failed (status {res.status}): {res.message}")

    charge    = res.x[:n]
    discharge = res.x[n:2*n]
    soc       = np.concatenate([[0.0], res.x[2*n:]])   # soc[0..n]

    revenue = float(np.dot(discharge, prices) - np.dot(charge, prices))
    return charge, discharge, soc[:n], revenue


# =============================================================================
# DATA
# =============================================================================

def get_available_years(filepath):
    """
    Return a sorted list of years with >= 8700 hours of data in the CSV.
    Reads only the date column so it is fast enough to use for UI population.
    """
    df = pd.read_csv(filepath, usecols=['Datetime (UTC)'])
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
    year_counts = df.groupby(df['Datetime (UTC)'].dt.year).size()
    return sorted(int(y) for y in year_counts[year_counts >= 8700].index)


def load_year_range(filepath, years, fx_rate=1.0, currency_symbol='EUR'):
    """
    Load all hours for the given list of years from a price CSV.
    Prices are converted from EUR using fx_rate.

    Returns: datetimes, prices, year_label (str), n_years (int)
    """
    df        = pd.read_csv(filepath)
    date_col  = 'Datetime (UTC)'
    price_col = 'Price (EUR/MWhe)'

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df_range = df[df[date_col].dt.year.isin(years)].copy().reset_index(drop=True)
    prices   = df_range[price_col].values.astype(float) * fx_rate

    nan_mask = np.isnan(prices)
    if nan_mask.any():
        prices = np.interp(np.arange(len(prices)),
                           np.where(~nan_mask)[0], prices[~nan_mask])

    n_years    = len(years)
    year_label = str(years[0]) if n_years == 1 else f'{years[0]}–{years[-1]}'

    print(f"Loaded {year_label}: {len(prices)} hours  "
          f"mean {currency_symbol}{prices.mean():.1f}/MWh  "
          f"std {prices.std():.1f}  "
          f"neg {(prices < 0).mean()*100:.1f}%")
    return df_range[date_col].values, prices, year_label, n_years


def load_most_recent_year(filepath, fx_rate=1.0, currency_symbol='EUR'):
    """Convenience wrapper for the CLI: loads the most recent complete year."""
    years = get_available_years(filepath)
    datetimes, prices, year_label, _ = load_year_range(
        filepath, [years[-1]], fx_rate, currency_symbol)
    return datetimes, prices, year_label


# =============================================================================
# CHART
# =============================================================================

def build_chart(datetimes, prices, year_label, country='Unknown',
                tech_config=None, currency_symbol='EUR', n_years=1, output=None):
    """
    Build and return a 2-panel Plotly figure.

    tech_config  : list of dicts, each with keys:
                     'duration' (int, hours), 'name' (str), 'rte' (float, 0–1)
                   Up to 4 entries. Defaults to DURATION_TECH values if None.
    currency_symbol : displayed on axis labels and hover text.
    n_years      : number of years in the analysis period, used to annualise
                   revenue figures shown in the legend and print output.
    output       : path to write an HTML file, or None to skip.

    Returns (fig, results) where results is a list of per-technology dicts
    containing the raw LP arrays (charge, discharge, soc) and summary values.
    """
    if tech_config is None:
        tech_config = [{'duration': d, **v} for d, v in DURATION_TECH.items()]

    dt_index      = pd.to_datetime(datetimes)
    results       = []
    interp_factor = int(round(1.0 / SOC_DISPLAY_STEP))
    ccy           = currency_symbol

    for tc in tech_config:
        d, rte, tech_name = tc['duration'], tc['rte'], tc['name']
        print(f"  LP {d}h ({tech_name} {rte*100:.0f}% RTE)...", end=' ', flush=True)
        charge, discharge, soc, revenue = lp_dispatch(prices, d, rte=rte)
        rev_ann = revenue / n_years
        print(f"{ccy}{rev_ann/1000:.1f}k/MW/yr  "
              f"({int(np.sum(discharge > 1e-4) / n_years)} discharge h/yr)")

        n_orig   = len(soc)
        n_fine   = (n_orig - 1) * interp_factor + 1
        soc_fine = np.interp(np.linspace(0, n_orig - 1, n_fine), np.arange(n_orig), soc)
        dt_fine  = pd.date_range(start=dt_index[0], periods=n_fine,
                                 freq=f'{int(SOC_DISPLAY_STEP * 60)}min')

        results.append(dict(duration=d, charge=charge, discharge=discharge,
                            soc=soc, soc_fine=soc_fine, dt_fine=dt_fine,
                            revenue=revenue, tech_name=tech_name, rte=rte))

    dur_label  = ' / '.join(f'{r["duration"]}h' for r in results)
    yr_suffix  = '' if n_years == 1 else f' (avg over {n_years} years)'

    # ── subplot: 2 rows ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.62],
        vertical_spacing=0.06,
        subplot_titles=[
            f'{country} Day-Ahead Electricity Price {year_label} ({ccy}/MWh)',
            f'State of Charge — {dur_label}, LP optimal',
        ]
    )

    # ── Row 1: price ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scattergl(
            x=dt_index, y=prices, mode='lines',
            name=f'Price ({ccy}/MWh)',
            line=dict(color='#475569', width=0.8),
            hovertemplate=f'%{{x|%d %b %H:%M}}<br>{ccy}%{{y:.1f}}/MWh<extra></extra>',
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line=dict(color='#94a3b8', width=1, dash='dot'), row=1, col=1)

    # ── Row 2: SoC ────────────────────────────────────────────────────────────
    max_dur = max(r['duration'] for r in results)

    for i, r in enumerate(results):
        colour, fill = PALETTE[i % len(PALETTE)]
        rev_ann_k = r['revenue'] / n_years / 1000
        d         = r['duration']

        fig.add_trace(
            go.Scatter(
                x=r['dt_fine'], y=r['soc_fine'],
                mode='lines',
                name=f'{d}h {r["tech_name"]} SoC  ({ccy}{rev_ann_k:.0f}k/MW/yr{yr_suffix})',
                line=dict(color=colour, width=1.6),
                fill='tozeroy', fillcolor=fill,
                hovertemplate=f'%{{x|%d %b %H:%M}}<br>{d}h SoC: %{{y:.2f}} h<extra></extra>',
            ),
            row=2, col=1
        )
        fig.add_hline(
            y=d,
            line=dict(color=colour, width=1, dash='dot'),
            annotation_text=f'{d}h max',
            annotation_position='top right',
            annotation_font=dict(color=colour, size=10),
            row=2, col=1,
        )

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f'Storage SoC — {country} {year_label} Day-Ahead Prices  '
                  f'({dur_label}, LP optimal)'),
            font=dict(size=15),
        ),
        height=780,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                    font=dict(size=10)),
        margin=dict(l=65, r=40, t=85, b=55),
        template='plotly_white',
    )

    fig.update_yaxes(title_text=f'{ccy}/MWh', row=1, col=1)
    fig.update_yaxes(title_text='SoC (hours, 1 MW system)',
                     range=[-0.1, max_dur + 0.5], row=2, col=1)

    fig.update_xaxes(
        title_text='Date',
        rangeslider=dict(visible=True, thickness=0.03),
        row=2, col=1,
    )
    # Rangeselector on row 2 avoids overlapping the row 1 subplot title
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=7,  label='1W',  step='day',   stepmode='backward'),
            dict(count=1,  label='1M',  step='month', stepmode='backward'),
            dict(count=3,  label='3M',  step='month', stepmode='backward'),
            dict(count=6,  label='6M',  step='month', stepmode='backward'),
            dict(step='all', label='All'),
        ]),
        row=2, col=1,
    )

    if output:
        fig.write_html(output, include_plotlyjs='cdn')
        print(f"\nChart saved: {output}")

    return fig, results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    filepath    = (sys.argv[1] if len(sys.argv) > 1
                   else 'european_wholesale_electricity_price_data_hourly/United Kingdom.csv')
    country     = os.path.splitext(os.path.basename(filepath))[0]
    tech_config = [{'duration': d, **v} for d, v in DURATION_TECH.items()]

    print(f"Loading: {filepath}")
    datetimes, prices, year_label = load_most_recent_year(
        filepath, fx_rate=EUR_TO_GBP, currency_symbol='GBP')

    print(f"\nRunning LP dispatch...")
    fig, _ = build_chart(datetimes, prices, year_label, country=country,
                         tech_config=tech_config, currency_symbol='GBP',
                         n_years=1, output='soc_lp_final.html')
