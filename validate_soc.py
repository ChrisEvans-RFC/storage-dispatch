"""
SOC Validation — Interactive Plot
==================================
Loads the most recent full year of UK wholesale price data, runs the
perfect-foresight dispatch for 4h and 12h durations, and produces an
interactive Plotly chart showing:
  - Hourly price (GBP/MWh)
  - State of Charge (%) for 4h storage
  - State of Charge (%) for 12h storage

Open the output HTML file in any browser to zoom / pan / hover.
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── mirror the same constants used in the main model ─────────────────────────
EUR_TO_GBP = 0.86

DISPATCH_WINDOW_MULTIPLIER = 4
DISPATCH_MAX_WINDOW = 336
DISPATCH_STEP = 24

TECHNOLOGIES = {
    'RFC':          {'rte': 0.75},
    'Li-Ion':       {'rte': 0.90},
    'Form Energy':  {'rte': 0.40},
}

# Per-duration technology assignments (RTE and label)
DURATION_TECH = {
    4:  {'name': 'LFP', 'rte': 0.85},
    12: {'name': 'RFC', 'rte': 0.75},
}
DEFAULT_TECH = {'name': 'RFC', 'rte': 0.75}

# Sub-hourly SOC display resolution (hours); 0.5 = 30-min interpolation
SOC_DISPLAY_STEP = 0.5


# ── dispatch returning SoC trace ──────────────────────────────────────────────

def dispatch_with_soc(prices, duration, rte=0.75, power_mw=1.0):
    """
    Same algorithm as optimal_dispatch_annual() but also returns the
    hourly SoC array (as a fraction 0-1 of max capacity).
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
            high_idx = sorted_idx[-(i + 1)]
            if low_idx == high_idx:
                continue
            if wp[high_idx] * rte > wp[low_idx]:
                charge_hours.add(int(low_idx))
                discharge_hours.add(int(high_idx))
            else:
                break

        overlap = charge_hours & discharge_hours
        charge_hours -= overlap
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

    soc_hours = soc[:n]  # MWh = hours of storage for a 1 MW system
    return dispatch, soc_hours


# ── load data ─────────────────────────────────────────────────────────────────

def load_most_recent_year(filepath):
    df = pd.read_csv(filepath)
    date_col = 'Datetime (UTC)'
    price_col = 'Price (EUR/MWhe)'

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Most recent complete year: last year with >= 8700 hours
    year_counts = df.groupby(df[date_col].dt.year).size()
    complete_years = year_counts[year_counts >= 8700].index
    target_year = int(complete_years.max())

    mask = df[date_col].dt.year == target_year
    df_year = df[mask].copy().reset_index(drop=True)

    prices_eur = df_year[price_col].values.astype(float)
    prices_gbp = prices_eur * EUR_TO_GBP

    # Interpolate any NaNs
    nan_mask = np.isnan(prices_gbp)
    if nan_mask.any():
        prices_gbp = np.interp(np.arange(len(prices_gbp)),
                               np.where(~nan_mask)[0],
                               prices_gbp[~nan_mask])

    datetimes = df_year[date_col].values
    print(f"Loaded {target_year}: {len(prices_gbp)} hours, "
          f"mean GBP {prices_gbp.mean():.1f}/MWh, "
          f"std {prices_gbp.std():.1f}, "
          f"negative {(prices_gbp < 0).mean()*100:.1f}%")
    return datetimes, prices_gbp, target_year


# ── build interactive chart ───────────────────────────────────────────────────

def build_chart(datetimes, prices, year, durations=(4, 12), output='soc_validation_v2.html'):
    dt_index = pd.to_datetime(datetimes)

    # Run dispatch for each duration
    results = {}
    interp_factor = int(round(1.0 / SOC_DISPLAY_STEP))  # e.g. 2 for 0.5h

    for d in durations:
        tech = DURATION_TECH.get(d, DEFAULT_TECH)
        rte  = tech['rte']
        tech_name = tech['name']
        print(f"  Dispatching {d}h {tech_name} (RTE={rte})...", end=' ', flush=True)
        dispatch, soc_hours = dispatch_with_soc(prices, duration=d, rte=rte)

        hourly_revenue = (
            np.where(dispatch > 0, dispatch * prices * rte, 0)
            + np.where(dispatch < 0, dispatch * prices, 0)
        )
        revenue = float(hourly_revenue.sum())

        # Cumulative series
        cum_discharge_h  = np.cumsum(dispatch > 0).astype(float)       # hours discharged
        cum_revenue_kw   = np.cumsum(hourly_revenue) / 1000             # £/kW (1 MW = 1000 kW)

        # Interpolate SoC to sub-hourly resolution for display
        n_orig = len(soc_hours)
        n_fine = (n_orig - 1) * interp_factor + 1
        soc_fine = np.interp(np.linspace(0, n_orig - 1, n_fine),
                             np.arange(n_orig), soc_hours)
        dt_fine = pd.date_range(start=dt_index[0], periods=n_fine,
                                freq=f'{int(SOC_DISPLAY_STEP * 60)}min')

        results[d] = {
            'dispatch': dispatch,
            'soc': soc_hours,
            'soc_fine': soc_fine,
            'dt_fine': dt_fine,
            'revenue': revenue,
            'tech_name': tech_name,
            'rte': rte,
            'cum_discharge_h': cum_discharge_h,
            'cum_revenue_kw': cum_revenue_kw,
        }
        cycles = int(np.sum(dispatch > 0))
        print(f"revenue GBP{revenue/1000:.0f}k/MW (GBP{revenue/1000:.0f}/kW), {cycles} discharge hours")

    # ── figure: 3 rows, shared x-axis; row 3 has secondary y-axis ───────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.36, 0.36],
        vertical_spacing=0.06,
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=[
            f'UK Wholesale Electricity Price {year} (GBP/MWh)',
            'State of Charge — 4h LFP (85% RTE) / 12h RFC (75% RTE)',
            'Cumulative Discharge Hours  &  Cumulative Revenue (£/kW)',
        ]
    )

    # Price trace (row 1)
    fig.add_trace(
        go.Scattergl(
            x=dt_index, y=prices,
            mode='lines',
            name='Price (GBP/MWh)',
            line=dict(color='#475569', width=0.8),
            hovertemplate='%{x|%d %b %H:%M}<br>£%{y:.1f}/MWh<extra></extra>',
        ),
        row=1, col=1
    )

    # Zero price line
    fig.add_hline(y=0, line=dict(color='#94a3b8', width=1, dash='dot'), row=1, col=1)

    # SoC traces (row 2) — one per duration
    colours = {4: '#2563eb', 12: '#16a34a', 24: '#dc2626', 8: '#d97706'}
    fill_colours = {4: 'rgba(37,99,235,0.12)', 12: 'rgba(22,163,74,0.12)',
                    24: 'rgba(220,38,38,0.12)', 8: 'rgba(217,119,6,0.12)'}
    default_colour = '#7c3aed'
    default_fill = 'rgba(124,58,237,0.12)'

    for d, res in results.items():
        colour = colours.get(d, default_colour)
        fill = fill_colours.get(d, default_fill)
        rev_k = res['revenue'] / 1000
        # Filled area under SoC curve (sub-hourly interpolated)
        fig.add_trace(
            go.Scatter(
                x=res['dt_fine'], y=res['soc_fine'],
                mode='lines',
                name=f'{d}h {res["tech_name"]} SoC  (£{rev_k:.0f}k/MW/yr)',
                line=dict(color=colour, width=1.5),
                fill='tozeroy',
                fillcolor=fill,
                hovertemplate=(
                    '%{x|%d %b %H:%M}<br>'
                    f'{d}h SoC: ' + '%{y:.2f} h<extra></extra>'
                ),
            ),
            row=2, col=1
        )
        # Capacity ceiling for this duration
        fig.add_hline(
            y=d, line=dict(color=colour, width=1, dash='dot'),
            annotation_text=f'{d}h max', annotation_position='top right',
            annotation_font=dict(color=colour, size=10),
            row=2, col=1
        )

    # Charge / discharge markers on the price panel for the first duration only
    first_d = list(results.keys())[0]
    first_colour = colours.get(first_d, default_colour)
    dispatch = results[first_d]['dispatch']
    charge_mask = dispatch < 0
    discharge_mask = dispatch > 0

    fig.add_trace(
        go.Scattergl(
            x=dt_index[charge_mask], y=prices[charge_mask],
            mode='markers',
            name=f'{first_d}h charging',
            marker=dict(color='rgba(37,99,235,0.5)', size=4, symbol='triangle-down'),
            hovertemplate='Charging<br>%{x|%d %b %H:%M}<br>£%{y:.1f}/MWh<extra></extra>',
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(
            x=dt_index[discharge_mask], y=prices[discharge_mask],
            mode='markers',
            name=f'{first_d}h discharging',
            marker=dict(color='rgba(220,38,38,0.5)', size=4, symbol='triangle-up'),
            hovertemplate='Discharging<br>%{x|%d %b %H:%M}<br>£%{y:.1f}/MWh<extra></extra>',
        ),
        row=1, col=1
    )

    # ── Row 3: cumulative discharge hours (left) + cumulative revenue/kW (right)
    for d, res in results.items():
        colour = colours.get(d, default_colour)
        rev_kw = res['revenue'] / 1000  # £/kW annual total

        # Left axis — cumulative discharge hours
        fig.add_trace(
            go.Scatter(
                x=dt_index, y=res['cum_discharge_h'],
                mode='lines',
                name=f'{d}h discharge hrs (total {res["cum_discharge_h"][-1]:.0f} h)',
                line=dict(color=colour, width=1.8),
                hovertemplate=(
                    '%{x|%d %b %H:%M}<br>'
                    f'{d}h cum. discharge: ' + '%{y:.0f} h<extra></extra>'
                ),
            ),
            row=3, col=1, secondary_y=False
        )

        # Right axis — cumulative revenue £/kW (dashed)
        fig.add_trace(
            go.Scatter(
                x=dt_index, y=res['cum_revenue_kw'],
                mode='lines',
                name=f'{d}h cum. revenue (£{rev_kw:.2f}/kW/yr)',
                line=dict(color=colour, width=1.8, dash='dash'),
                hovertemplate=(
                    '%{x|%d %b %H:%M}<br>'
                    f'{d}h cum. revenue: £' + '%{y:.2f}/kW<extra></extra>'
                ),
            ),
            row=3, col=1, secondary_y=True
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=f'Storage Dispatch & SoC Validation — UK {year} Real Prices (4h LFP 85% / 12h RFC 75%)',
            font=dict(size=16),
        ),
        height=980,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=10)),
        margin=dict(l=70, r=70, t=90, b=60),
        template='plotly_white',
    )

    max_dur = max(durations)
    fig.update_yaxes(title_text='GBP/MWh', row=1, col=1)
    fig.update_yaxes(title_text='SoC (hours, 1 MW system)', range=[-0.1, max_dur + 0.5], row=2, col=1)
    fig.update_yaxes(title_text='Cumulative discharge (hours)',
                     secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text='Cumulative revenue (£/kW)',
                     secondary_y=True, row=3, col=1)

    fig.update_xaxes(
        title_text='Date',
        rangeslider=dict(visible=True, thickness=0.03),
        row=3, col=1
    )

    # Range selector buttons on the price panel
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=7,  label='1W',  step='day',   stepmode='backward'),
                dict(count=1,  label='1M',  step='month', stepmode='backward'),
                dict(count=3,  label='3M',  step='month', stepmode='backward'),
                dict(count=6,  label='6M',  step='month', stepmode='backward'),
                dict(step='all', label='All'),
            ]
        ),
        row=1, col=1
    )

    fig.write_html(output, include_plotlyjs='cdn')
    print(f"\nInteractive chart saved: {output}")
    print("Open this file in your browser to zoom, pan, and hover.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    filepath = (sys.argv[1] if len(sys.argv) > 1
                else 'european_wholesale_electricity_price_data_hourly/United Kingdom.csv')

    # Optional: override durations via command line, e.g. "4,12,24"
    if len(sys.argv) > 2:
        durations = tuple(int(x) for x in sys.argv[2].split(','))
    else:
        durations = (4, 12)

    print(f"Loading: {filepath}")
    datetimes, prices, year = load_most_recent_year(filepath)

    print(f"\nRunning dispatch for durations: {durations}")
    build_chart(datetimes, prices, year, durations=durations, output='soc_validation_v2.html')
