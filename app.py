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
from soc_lp_final import load_most_recent_year, build_chart

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

    with st.spinner(f'Loading {country} price data...'):
        datetimes, prices, year = load_most_recent_year(
            filepath, fx_rate=fx_rate, currency_symbol=currency_symbol
        )

    st.info(
        f'**{country} {year}** — {len(prices):,} hours loaded  |  '
        f'Mean {currency_symbol}{prices.mean():.1f}/MWh  |  '
        f'Negative hours: {(prices < 0).mean()*100:.1f}%'
    )

    with st.spinner('Running LP optimisation (this may take a few seconds)...'):
        fig, results = build_chart(
            datetimes, prices, year,
            country=country,
            tech_config=tech_config,
            currency_symbol=currency_symbol,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
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
        cycles           = discharge_energy / duration if duration > 0 else 0.0
        avg_charge_price = charge_cost    / charge_energy    if charge_energy    > 0 else 0.0
        avg_disc_price   = discharge_rev  / discharge_energy if discharge_energy > 0 else 0.0

        rows.append({
            'Technology':                          r['tech_name'],
            'Duration (h)':                        duration,
            'RTE (%)':                             int(r['rte'] * 100),
            f'Net revenue ({currency_symbol}k/MW/yr)': round(net_revenue / 1000, 1),
            f'Discharge revenue ({currency_symbol}k/MW/yr)': round(discharge_rev / 1000, 1),
            f'Charge cost ({currency_symbol}k/MW/yr)':       round(charge_cost   / 1000, 1),
            'Discharge hours/yr':                  int(np.sum(d_mask)),
            'Charge hours/yr':                     int(np.sum(c_mask)),
            'Discharge energy (MWh/yr)':           round(discharge_energy, 0),
            'Charge energy (MWh/yr)':              round(charge_energy,    0),
            'Cycles/yr':                           round(cycles, 1),
            f'Avg discharge price ({currency_symbol}/MWh)': round(avg_disc_price,   1),
            f'Avg charge price ({currency_symbol}/MWh)':    round(avg_charge_price, 1),
            f'Price spread captured ({currency_symbol}/MWh)': round(avg_disc_price - avg_charge_price, 1),
        })

    df_summary = pd.DataFrame(rows).set_index('Technology')
    st.dataframe(df_summary, use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────────
    html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
    st.download_button(
        label='Download chart as HTML',
        data=html_bytes,
        file_name=f'storage_dispatch_{country.replace(" ", "_")}_{year}.html',
        mime='text/html',
    )

else:
    st.info('Configure settings in the sidebar and click **Run optimisation**.')
