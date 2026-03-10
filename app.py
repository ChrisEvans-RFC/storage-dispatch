"""
Storage Dispatch Optimiser — Streamlit App
==========================================
Interactive front-end for soc_lp_final.py.
Run locally with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from soc_lp_final import get_available_years, load_year_range, build_chart

DATA_DIR = 'european_wholesale_electricity_price_data_hourly'

CURRENCIES = {
    'GBP (£)': ('GBP', 0.86),
    'EUR (€)': ('EUR', 1.00),
    'USD ($)': ('USD', 1.08),
}

# Defaults shown in the technology rows
TECH_DEFAULTS = [
    (4,   '4h LFP',    85),
    (12,  '12h RFC',   75),
    (100, '100h LDES', 40),
    (24,  '24h',       70),
]

# Preset period options (number of years, in order of preference)
PRESET_NS = [1, 2, 3, 5, 10]


@st.cache_data
def cached_available_years(filepath):
    return get_available_years(filepath)


st.set_page_config(page_title='Storage Dispatch Optimiser', layout='wide')
st.title('Storage Dispatch Optimiser')
st.caption(
    'LP-optimal perfect-foresight dispatch on European day-ahead prices. '
    'Results assume a 1 MW / N MWh system with no degradation or fixed costs.'
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header('Settings')

    # ── Country ───────────────────────────────────────────────────────────────
    countries = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f != 'all_countries.csv'
    ])
    default_idx = countries.index('United Kingdom') if 'United Kingdom' in countries else 0
    country = st.selectbox('Country', countries, index=default_idx)

    # ── Analysis period ───────────────────────────────────────────────────────
    st.divider()
    filepath_sidebar = os.path.join(DATA_DIR, f'{country}.csv')
    avail_years      = cached_available_years(filepath_sidebar)
    n_avail          = len(avail_years)

    period_options = []
    for n in PRESET_NS:
        if n <= n_avail:
            yr_start = avail_years[-n]
            yr_end   = avail_years[-1]
            yr_range = str(yr_end) if n == 1 else f'{yr_start}–{yr_end}'
            label    = 'Most recent year' if n == 1 else f'Last {n} years'
            period_options.append((f'{label} ({yr_range})', avail_years[-n:]))

    # Add "All available" if n_avail isn't already one of the presets
    if n_avail > 0 and n_avail not in PRESET_NS:
        yr_range = str(avail_years[0]) if n_avail == 1 else f'{avail_years[0]}–{avail_years[-1]}'
        period_options.append(
            (f'All available ({yr_range}, {n_avail} yrs)', avail_years)
        )

    period_labels  = [o[0] for o in period_options]
    selected_label = st.selectbox('Analysis period', period_labels)
    selected_years = next(o[1] for o in period_options if o[0] == selected_label)
    n_years        = len(selected_years)

    # ── Currency ──────────────────────────────────────────────────────────────
    st.divider()
    currency_label = st.selectbox('Display currency', list(CURRENCIES.keys()))
    currency_symbol, default_fx = CURRENCIES[currency_label]
    fx_rate = st.number_input(
        f'EUR → {currency_symbol} exchange rate',
        value=float(default_fx), min_value=0.01, step=0.01, format='%.4f'
    )

    # ── Technologies ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader('Storage technologies')
    n_techs = st.number_input(
        'Number of technologies (max 4)', min_value=1, max_value=4, value=3, step=1
    )

    tech_config = []
    for i in range(n_techs):
        d_def, name_def, rte_def = TECH_DEFAULTS[i]
        with st.expander(f'Technology {i + 1}', expanded=True):
            duration = st.number_input(
                'Duration (hours)', min_value=1, max_value=8760,
                value=d_def, step=1, key=f'dur_{i}'
            )
            name = st.text_input('Label', value=name_def, key=f'name_{i}')
            rte_pct = st.slider(
                'Round-trip efficiency (%)', min_value=10, max_value=100,
                value=rte_def, step=5, key=f'rte_{i}'
            )
            tech_config.append({
                'duration': int(duration),
                'name':     name,
                'rte':      rte_pct / 100,
            })

    st.divider()
    run = st.button('Run optimisation', type='primary', use_container_width=True)

# =============================================================================
# MAIN AREA
# =============================================================================

if run:
    filepath = os.path.join(DATA_DIR, f'{country}.csv')

    with st.spinner(f'Loading {country} price data ({selected_label})...'):
        datetimes, prices, year_label, n_years = load_year_range(
            filepath, selected_years, fx_rate=fx_rate, currency_symbol=currency_symbol
        )

    ann_note = '' if n_years == 1 else f'  |  Metrics shown as annual averages over {n_years} years'
    st.info(
        f'**{country} {year_label}** — {len(prices):,} hours loaded  |  '
        f'Mean {currency_symbol}{prices.mean():.1f}/MWh  |  '
        f'Negative hours: {(prices < 0).mean()*100:.1f}%'
        + ann_note
    )

    spinner_msg = ('Running LP optimisation — this covers '
                   f'{len(prices):,} hours so may take up to a minute...'
                   if n_years > 1 else
                   'Running LP optimisation (this may take a few seconds)...')

    with st.spinner(spinner_msg):
        fig, results = build_chart(
            datetimes, prices, year_label,
            country=country,
            tech_config=tech_config,
            currency_symbol=currency_symbol,
            n_years=n_years,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    ann = f' (avg/yr)' if n_years > 1 else '/yr'
    st.subheader('Performance summary')

    rows = []
    for r in results:
        charge    = r['charge']
        discharge = r['discharge']
        duration  = r['duration']

        c_mask = charge    > 1e-4
        d_mask = discharge > 1e-4

        charge_energy    = float(np.sum(charge))
        discharge_energy = float(np.sum(discharge))
        charge_cost      = float(np.dot(charge,    prices))
        discharge_rev    = float(np.dot(discharge, prices))
        net_revenue      = discharge_rev - charge_cost
        avg_charge_price = charge_cost    / charge_energy    if charge_energy    > 0 else 0.0
        avg_disc_price   = discharge_rev  / discharge_energy if discharge_energy > 0 else 0.0

        rows.append({
            'Technology':                                      r['tech_name'],
            'Duration (h)':                                    duration,
            'RTE (%)':                                         int(r['rte'] * 100),
            f'Net revenue ({currency_symbol}k/MW{ann})':      round(net_revenue    / n_years / 1000, 1),
            f'Discharge revenue ({currency_symbol}k/MW{ann})':round(discharge_rev  / n_years / 1000, 1),
            f'Charge cost ({currency_symbol}k/MW{ann})':      round(charge_cost    / n_years / 1000, 1),
            f'Discharge hours{ann}':                           round(np.sum(d_mask) / n_years, 0),
            f'Charge hours{ann}':                              round(np.sum(c_mask) / n_years, 0),
            f'Discharge energy (MWh{ann})':                   round(discharge_energy / n_years, 0),
            f'Charge energy (MWh{ann})':                      round(charge_energy    / n_years, 0),
            f'Cycles{ann}':                                    round(discharge_energy / n_years / duration, 1),
            f'Avg discharge price ({currency_symbol}/MWh)':   round(avg_disc_price,   1),
            f'Avg charge price ({currency_symbol}/MWh)':      round(avg_charge_price, 1),
            f'Price spread ({currency_symbol}/MWh)':          round(avg_disc_price - avg_charge_price, 1),
        })

    df_summary = pd.DataFrame(rows).set_index('Technology')
    st.dataframe(df_summary, use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────────
    safe_label = year_label.replace('–', '-')
    html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
    st.download_button(
        label='Download chart as HTML',
        data=html_bytes,
        file_name=f'storage_dispatch_{country.replace(" ", "_")}_{safe_label}.html',
        mime='text/html',
    )

else:
    st.info('Configure settings in the sidebar and click **Run optimisation**.')
