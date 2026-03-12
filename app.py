"""
Storage Dispatch Optimiser
==========================
Single-page layout:
  1. SOC time-history chart (LP-optimal dispatch for configured technologies)
  2. Performance summary table
  3. Sensitivity: metric contour / lines across the duration × efficiency space

Run with:  streamlit run app.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from soc_lp_final import get_available_years, load_year_range, lp_dispatch, build_chart

DATA_DIR = 'european_wholesale_electricity_price_data_hourly'

# ── constants ──────────────────────────────────────────────────────────────────

CURRENCIES = {
    'GBP (£)': ('GBP', '£', 0.86),
    'EUR (€)': ('EUR', '€', 1.00),
    'USD ($)': ('USD', '$', 1.08),
}

PRESET_NS = [1, 2, 3, 5, 10]

TECH_DEFAULTS = [
    (4,   'LFP',       85,  600),
    (12,  'RFC Power', 75, 1000),
    (100, 'Iron-Air',  40, 1000),
    (24,  '',          70,    0),
]

TS_DURATIONS        = [2, 4, 8, 16, 32, 64, 128]
TS_EFFICIENCIES_PCT = list(range(30, 110, 10))
TS_EFFICIENCIES     = [e / 100 for e in TS_EFFICIENCIES_PCT]

TECH_MARKERS = [
    {'name': 'LFP',       'duration':  4,  'rte': 0.85, 'symbol': 'square',  'color': '#2563eb'},
    {'name': 'RFC Power', 'duration': 12,  'rte': 0.75, 'symbol': 'diamond', 'color': '#16a34a'},
    {'name': 'Iron-Air',  'duration': 100, 'rte': 0.40, 'symbol': 'circle',  'color': '#dc2626'},
]

METRIC_OPTIONS = [
    'Net revenue (k{ccy}/MW/yr)',
    'Cycles per year',
    'Avg discharge price ({ccy}/MWh)',
    'Avg charge price ({ccy}/MWh)',
    'Price spread ({ccy}/MWh)',
    'Discharge hours per year',
    'Utilisation (%)',
]

METRIC_COLORSCALE = {
    'Net revenue (k{ccy}/MW/yr)':        'RdYlGn',
    'Cycles per year':                    'Viridis',
    'Avg discharge price ({ccy}/MWh)':    'Plasma',
    'Avg charge price ({ccy}/MWh)':       'Plasma_r',
    'Price spread ({ccy}/MWh)':           'RdYlGn',
    'Discharge hours per year':           'Viridis',
    'Utilisation (%)':                    'Viridis',
}


# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_available_years(filepath):
    return get_available_years(filepath)


def _dispatch_one(prices, dur, rte):
    charge, discharge, _soc, _rev = lp_dispatch(prices, dur, rte=rte)
    de = float(np.sum(discharge))
    ce = float(np.sum(charge))
    dh = int(np.sum(discharge > 1e-4))
    dr = float(np.dot(discharge, prices))
    cc = float(np.dot(charge,    prices))
    return {'net_revenue': dr - cc, 'discharge_rev': dr, 'charge_cost': cc,
            'discharge_energy': de, 'charge_energy': ce, 'discharge_hours': dh}


@st.cache_data(show_spinner=False)
def _run_dispatch(prices_bytes: bytes, dur: int, rte_milli: int) -> dict:
    """Single cached LP run keyed on (prices, duration, rte×1000)."""
    prices = np.frombuffer(prices_bytes, dtype=np.float64).copy()
    return _dispatch_one(prices, dur, rte_milli / 1000)


def _assemble_grids(raw: dict, n_years: int, n_hours: int) -> dict:
    n_eff = len(TS_EFFICIENCIES)
    n_dur = len(TS_DURATIONS)
    grids = {k: np.zeros((n_eff, n_dur)) for k in [
        'net_revenue', 'cycles', 'avg_discharge_price', 'avg_charge_price',
        'price_spread', 'discharge_hours', 'utilisation',
    ]}
    for (i, j), r in raw.items():
        de  = r['discharge_energy']
        ce  = r['charge_energy']
        cap = float(TS_DURATIONS[j])
        dh  = r['discharge_hours']
        grids['net_revenue'][i, j]         = r['net_revenue']          / n_years / 1e3
        grids['cycles'][i, j]              = (de / n_years / cap)      if cap > 0 else 0.0
        grids['avg_discharge_price'][i, j] = (r['discharge_rev'] / de) if de  > 0 else 0.0
        grids['avg_charge_price'][i, j]    = (r['charge_cost']    / ce) if ce  > 0 else 0.0
        grids['price_spread'][i, j]        = (
            (r['discharge_rev'] / de if de > 0 else 0.0) -
            (r['charge_cost']    / ce if ce > 0 else 0.0)
        )
        grids['discharge_hours'][i, j]     = dh / n_years
        grids['utilisation'][i, j]         = dh / n_hours * 100.0
    return grids


def _interp_grid(z, dur, rte_pct):
    """Bilinear interpolation in log-duration / linear-efficiency space."""
    log_durs = np.log10(TS_DURATIONS)
    j_f = float(np.interp(np.log10(dur), log_durs, np.arange(len(TS_DURATIONS))))
    i_f = float(np.interp(rte_pct, TS_EFFICIENCIES_PCT, np.arange(len(TS_EFFICIENCIES_PCT))))
    j0  = max(0, min(int(j_f), len(TS_DURATIONS) - 2))
    i0  = max(0, min(int(i_f), len(TS_EFFICIENCIES_PCT) - 2))
    j1, i1 = j0 + 1, i0 + 1
    dj, di = j_f - j0, i_f - i0
    return (z[i0, j0] * (1-di) * (1-dj) + z[i0, j1] * (1-di) * dj +
            z[i1, j0] * di     * (1-dj) + z[i1, j1] * di     * dj)


def _metric_key(tmpl):
    return {
        'Net revenue (k{ccy}/MW/yr)':      'net_revenue',
        'Cycles per year':                  'cycles',
        'Avg discharge price ({ccy}/MWh)': 'avg_discharge_price',
        'Avg charge price ({ccy}/MWh)':    'avg_charge_price',
        'Price spread ({ccy}/MWh)':        'price_spread',
        'Discharge hours per year':         'discharge_hours',
        'Utilisation (%)':                  'utilisation',
    }[tmpl]


def _ts_axis(log_x):
    x_vals    = np.log10(TS_DURATIONS) if log_x else TS_DURATIONS
    tick_vals = x_vals
    tick_text = [str(d) for d in TS_DURATIONS]
    title     = 'Duration (hours, log scale)' if log_x else 'Duration (hours)'
    return x_vals, tick_vals, tick_text, title


def build_contour(grids, metric_tmpl, ccy_sym, year_label, country, log_x=True):
    key    = _metric_key(metric_tmpl)
    z      = grids[key]
    label  = metric_tmpl.replace('{ccy}', ccy_sym)
    cscale = METRIC_COLORSCALE[metric_tmpl]
    zmid   = 0.0 if (z.min() < 0 < z.max()) and key in ('net_revenue', 'price_spread') else None
    x_vals, tick_vals, tick_text, x_title = _ts_axis(log_x)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_vals, y=TS_EFFICIENCIES_PCT, z=z,
        colorscale=cscale, zmid=zmid,
        contours=dict(coloring='heatmap', showlabels=True,
                      labelfont=dict(size=10, color='white')),
        colorbar=dict(title=dict(text=label, side='right')),
        hovertemplate='Duration: %{x}<br>Efficiency: %{y}%<br>' + label + ': %{z:.2f}<extra></extra>',
        name=label,
    ))
    for tm in TECH_MARKERS:
        x_mark  = np.log10(tm['duration']) if log_x else tm['duration']
        val_str = f'{_interp_grid(z, tm["duration"], tm["rte"]*100):.2f}'
        fig.add_trace(go.Scatter(
            x=[x_mark], y=[tm['rte'] * 100], mode='markers+text', name=tm['name'],
            marker=dict(symbol=tm['symbol'], size=14, color=tm['color'],
                        line=dict(width=2, color='white')),
            text=[tm['name']], textposition='top center',
            textfont=dict(size=11, color=tm['color']),
            hovertemplate=(f"<b>{tm['name']}</b><br>Duration: {tm['duration']}h<br>"
                           f"RTE: {int(tm['rte']*100)}%<br>{label}: {val_str}<extra></extra>"),
        ))
    fig.update_layout(
        title=dict(text=f'Sensitivity — {label}<br>'
                        f'<sup>{country} {year_label} · LP perfect-foresight dispatch</sup>',
                   font=dict(size=16)),
        xaxis=dict(title=x_title, tickvals=tick_vals, ticktext=tick_text),
        yaxis=dict(title='Round-trip efficiency (%)'),
        height=550, template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.04, x=0, font=dict(size=11)),
        margin=dict(l=65, r=30, t=110, b=60),
    )
    return fig


def build_lines(grids, metric_tmpl, ccy_sym, year_label, country, log_x=True):
    key    = _metric_key(metric_tmpl)
    z      = grids[key]
    label  = metric_tmpl.replace('{ccy}', ccy_sym)
    n_eff  = len(TS_EFFICIENCIES_PCT)
    colors = [f'hsl({int(240 - 240 * i / max(n_eff - 1, 1))},80%,45%)' for i in range(n_eff)]
    x_vals, tick_vals, tick_text, x_title = _ts_axis(log_x)

    fig = go.Figure()
    for i, (eff_pct, color) in enumerate(zip(TS_EFFICIENCIES_PCT, colors)):
        fig.add_trace(go.Scatter(
            x=x_vals, y=z[i], mode='lines+markers', name=f'{eff_pct}% RTE',
            line=dict(color=color, width=2), marker=dict(size=5),
            hovertemplate=(f'<b>{eff_pct}% RTE</b><br>Duration: %{{text}}h<br>'
                           + label + ': %{y:.2f}<extra></extra>'),
            text=[str(d) for d in TS_DURATIONS],
        ))

    # Technology markers — interpolated values at exact (duration, rte) coordinates
    for tm in TECH_MARKERS:
        x_mark = np.log10(tm['duration']) if log_x else tm['duration']
        y_mark = _interp_grid(z, tm['duration'], tm['rte'] * 100)
        fig.add_trace(go.Scatter(
            x=[x_mark], y=[y_mark],
            mode='markers+text',
            name=tm['name'],
            marker=dict(symbol=tm['symbol'], size=14, color=tm['color'],
                        line=dict(width=2, color='white')),
            text=[f"{tm['name']}<br>{y_mark:.1f}"],
            textposition='top center',
            textfont=dict(size=11, color=tm['color']),
            hovertemplate=(f"<b>{tm['name']}</b><br>Duration: {tm['duration']}h<br>"
                           f"RTE: {int(tm['rte']*100)}%<br>"
                           f"{label}: {y_mark:.2f}<extra></extra>"),
            showlegend=True,
        ))

    fig.add_hline(y=0, line=dict(color='#94a3b8', width=1, dash='dot'))
    fig.update_layout(
        title=dict(text=f'{label} vs Duration<br>'
                        f'<sup>{country} {year_label} · LP perfect-foresight dispatch</sup>',
                   font=dict(size=16)),
        xaxis=dict(title=x_title, tickvals=tick_vals, ticktext=tick_text),
        yaxis=dict(title=label),
        height=550, template='plotly_white',
        legend=dict(title='Efficiency / Technology', orientation='v',
                    x=1.02, y=1, font=dict(size=10)),
        margin=dict(l=65, r=160, t=110, b=60),
        hovermode='x unified',
    )
    return fig


# ── page ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Storage Dispatch Optimiser', layout='wide')
st.title('Storage Dispatch Optimiser')
st.caption(
    'LP-optimal perfect-foresight dispatch on European day-ahead prices. '
    '1 MW / N MWh system — no degradation or fixed costs.'
)

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Settings')

    countries   = sorted([os.path.splitext(f)[0] for f in os.listdir(DATA_DIR)
                          if f.endswith('.csv') and f != 'all_countries.csv'])
    default_idx = countries.index('United Kingdom') if 'United Kingdom' in countries else 0
    country     = st.selectbox('Country', countries, index=default_idx)

    st.divider()
    avail_yrs = cached_available_years(os.path.join(DATA_DIR, f'{country}.csv'))
    n_avail   = len(avail_yrs)
    period_options = []
    for n in PRESET_NS:
        if n <= n_avail:
            yr_end   = avail_yrs[-1]
            yr_start = avail_yrs[-n]
            yr_range = str(yr_end) if n == 1 else f'{yr_start}–{yr_end}'
            lbl      = 'Most recent year' if n == 1 else f'Last {n} years'
            period_options.append((f'{lbl} ({yr_range})', avail_yrs[-n:]))
    if n_avail > 0 and n_avail not in PRESET_NS:
        yr_range = str(avail_yrs[0]) if n_avail == 1 else f'{avail_yrs[0]}–{avail_yrs[-1]}'
        period_options.append((f'All available ({yr_range}, {n_avail} yrs)', avail_yrs))
    selected_label = st.selectbox('Analysis period', [o[0] for o in period_options])
    selected_years = next(o[1] for o in period_options if o[0] == selected_label)
    n_years        = len(selected_years)

    st.divider()
    ccy_label             = st.selectbox('Display currency', list(CURRENCIES.keys()))
    ccy_code, ccy_sym, default_fx = CURRENCIES[ccy_label]
    fx_rate = st.number_input(f'EUR → {ccy_code} exchange rate',
                               value=float(default_fx), min_value=0.01,
                               step=0.01, format='%.4f')

    st.divider()
    st.subheader('Storage technologies')
    n_techs     = st.number_input('Number of technologies (max 4)',
                                   min_value=1, max_value=4, value=3, step=1)
    tech_config = []
    for i in range(n_techs):
        d_def, name_def, rte_def, cyc_def = TECH_DEFAULTS[i]
        with st.expander(f'Technology {i + 1}', expanded=True):
            c1, c2 = st.columns([3, 2])
            name     = c1.text_input('Label', value=name_def, key=f'name_{i}')
            duration = c2.number_input('Duration (h)', min_value=1, max_value=8760,
                                        value=d_def, step=1, key=f'dur_{i}')
            rte_pct  = st.slider('RTE (%)', min_value=10, max_value=100,
                                  value=rte_def, step=5, key=f'rte_{i}')
            max_cyc  = st.number_input('Max cycles/yr  (0 = unlimited)',
                                        min_value=0, value=cyc_def, step=50, key=f'cyc_{i}')
            tech_config.append({'duration': int(duration), 'name': name,
                                 'rte': rte_pct / 100, 'max_cycles': int(max_cyc)})

    st.divider()
    run = st.button('Run analysis', type='primary', use_container_width=True)
    n_grid = len(TS_DURATIONS) * len(TS_EFFICIENCIES)
    st.caption(f'{n_techs} SOC dispatch{"es" if n_techs > 1 else ""} + '
               f'{n_grid} sensitivity LP runs. Cached after first run.')

# ── cache key ─────────────────────────────────────────────────────────────────

_cache_key = (
    country, tuple(selected_years), fx_rate,
    tuple((t['duration'], t['rte'], t['max_cycles']) for t in tech_config),
    len(TS_DURATIONS), len(TS_EFFICIENCIES),
)

# ── compute ───────────────────────────────────────────────────────────────────

if run:
    filepath     = os.path.join(DATA_DIR, f'{country}.csv')
    prices_bytes = None  # set after load

    progress = st.progress(0, text='Loading price data…')

    datetimes, prices, year_label, n_yrs = load_year_range(
        filepath, selected_years, fx_rate=fx_rate, currency_symbol=ccy_sym,
    )
    prices_bytes = prices.tobytes()
    progress.progress(5, text='Running SOC dispatch for configured technologies…')

    fig_soc, soc_results = build_chart(
        datetimes, prices, year_label,
        country=country, tech_config=tech_config,
        currency_symbol=ccy_sym, n_years=n_yrs,
    )
    progress.progress(20, text=f'Running {n_grid} sensitivity LP optimisations…')

    # Sensitivity grid — parallel, with per-future progress updates
    tasks = [(i, j, int(TS_EFFICIENCIES[i] * 1000), TS_DURATIONS[j])
             for i in range(len(TS_EFFICIENCIES))
             for j in range(len(TS_DURATIONS))]
    raw: dict = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        fmap = {ex.submit(_run_dispatch, prices_bytes, dur, rte_milli): (i, j)
                for i, j, rte_milli, dur in tasks}
        for k, fut in enumerate(as_completed(fmap)):
            i, j = fmap[fut]
            raw[(i, j)] = fut.result()
            pct = 20 + int(78 * (k + 1) / n_grid)
            progress.progress(pct, text=f'Sensitivity: {k+1}/{n_grid} LP runs complete…')

    grids = _assemble_grids(raw, n_yrs, len(prices))
    progress.progress(100, text='Done!')
    progress.empty()

    st.session_state.update({
        'cache_key':   _cache_key,
        'fig_soc':     fig_soc,
        'soc_results': soc_results,
        'tech_config': tech_config,
        'prices':      prices,
        'year_label':  year_label,
        'n_yrs':       n_yrs,
        'grids':       grids,
        'price_info': (
            f'**{country} {year_label}** — {len(prices):,} hours  |  '
            f'Mean {ccy_sym}{prices.mean():.1f}/MWh  |  '
            f'Std {prices.std():.1f}  |  '
            f'Neg. {(prices < 0).mean()*100:.1f}%'
            + (f'  |  avg over {n_yrs} years' if n_yrs > 1 else '')
        ),
    })

# ── render ────────────────────────────────────────────────────────────────────

if ('cache_key' in st.session_state
        and st.session_state['cache_key'] == _cache_key):

    fig_soc     = st.session_state['fig_soc']
    soc_results = st.session_state['soc_results']
    tech_config = st.session_state['tech_config']
    prices      = st.session_state['prices']
    year_label  = st.session_state['year_label']
    n_yrs       = st.session_state['n_yrs']
    grids       = st.session_state['grids']

    st.info(st.session_state['price_info'])

    # ── 1. SOC time history ────────────────────────────────────────────────────
    st.plotly_chart(fig_soc, use_container_width=True)

    # ── 2. Performance summary table ───────────────────────────────────────────
    ann = ' (avg/yr)' if n_yrs > 1 else '/yr'
    st.subheader('Performance summary')
    rows = []
    for r, tc in zip(soc_results, tech_config):
        charge    = r['charge']
        discharge = r['discharge']
        duration  = r['duration']
        de        = float(np.sum(discharge))
        ce        = float(np.sum(charge))
        cc        = float(np.dot(charge,    prices))
        dr        = float(np.dot(discharge, prices))
        net_rev   = (dr - cc) / n_yrs
        avg_dp    = dr / de if de > 0 else 0.0
        avg_cp    = cc / ce if ce > 0 else 0.0
        max_cyc   = tc['max_cycles']
        cyc_cap   = str(max_cyc) if max_cyc > 0 else '—'
        rows.append({
            'Technology':                            r['tech_name'],
            'Duration (h)':                          duration,
            'RTE (%)':                               int(r['rte'] * 100),
            'Max cycles/yr cap':                     cyc_cap,
            f'Net revenue ({ccy_sym}k/MW{ann})':    round(net_rev / 1000, 1),
            f'Discharge rev. ({ccy_sym}k/MW{ann})': round(dr / n_yrs / 1000, 1),
            f'Charge cost ({ccy_sym}k/MW{ann})':    round(cc / n_yrs / 1000, 1),
            f'Discharge h{ann}':                     round(np.sum(discharge > 1e-4) / n_yrs, 0),
            f'Charge h{ann}':                        round(np.sum(charge    > 1e-4) / n_yrs, 0),
            f'Cycles{ann}':                          round(de / n_yrs / duration, 1),
            f'Avg disc. price ({ccy_sym}/MWh)':     round(avg_dp, 1),
            f'Avg chg. price ({ccy_sym}/MWh)':      round(avg_cp, 1),
            f'Spread ({ccy_sym}/MWh)':              round(avg_dp - avg_cp, 1),
        })
    st.dataframe(pd.DataFrame(rows).set_index('Technology'), use_container_width=True)

    # ── 3. Sensitivity ─────────────────────────────────────────────────────────
    st.subheader('Sensitivity — duration × efficiency space')
    metric_labels = [m.replace('{ccy}', ccy_sym) for m in METRIC_OPTIONS]
    ctrl_col, toggle_col = st.columns([4, 1])
    with ctrl_col:
        selected_display = st.radio('Metric', metric_labels, horizontal=True)
    with toggle_col:
        view  = st.radio('View', ['Contour', 'Lines'], horizontal=True)
        log_x = st.checkbox('Log x-axis', value=True)

    selected_tmpl = METRIC_OPTIONS[metric_labels.index(selected_display)]
    if view == 'Contour':
        fig_ts = build_contour(grids, selected_tmpl, ccy_sym, year_label, country, log_x)
    else:
        fig_ts = build_lines(grids, selected_tmpl, ccy_sym, year_label, country, log_x)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── 4. Downloads ───────────────────────────────────────────────────────────
    safe_label = year_label.replace('–', '-')
    st.download_button(
        label='Download SOC chart (HTML)',
        data=fig_soc.to_html(include_plotlyjs='cdn').encode(),
        file_name=f'soc_{country.replace(" ","_")}_{safe_label}.html',
        mime='text/html',
    )
    with st.expander('Download sensitivity grid (CSV)'):
        dl_cols = st.columns(2)
        for k_idx, metric_tmpl in enumerate(METRIC_OPTIONS):
            key   = _metric_key(metric_tmpl)
            m_lbl = metric_tmpl.replace('{ccy}', ccy_sym)
            df_out = pd.DataFrame(
                grids[key],
                index=pd.Index(TS_EFFICIENCIES_PCT, name='RTE (%)'),
                columns=pd.Index(TS_DURATIONS, name='Duration (h)'),
            )
            safe_name = (m_lbl.replace('/', '_per_').replace(' ', '_')
                              .replace('(', '').replace(')', '').lower())
            dl_cols[k_idx % 2].download_button(
                label=f'Download: {m_lbl}',
                data=df_out.to_csv().encode(),
                file_name=f'sensitivity_{country.replace(" ","_")}_{safe_label}_{safe_name}.csv',
                mime='text/csv',
                key=f'dl_{key}',
            )

else:
    st.info('Configure settings in the sidebar and click **Run analysis**.')
