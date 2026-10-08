"""
Print-quality export of the two time-based panels (price and state of charge).

The interactive Plotly figure is sized in screen pixels, so a browser
screenshot of it drops into Word with text far below readable size. This module
re-renders the same data with matplotlib on a canvas measured in *inches* and
text measured in *points*: place the image in Word at its natural width and the
labels come out at exactly the point sizes set here.

Entry point: render_time_charts(...) -> bytes
"""

import io

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soc_lp_final import PALETTE

# Usable text width for common page setups, in inches.
WORD_WIDTHS = {
    'Full page (6.5 in)':  6.5,
    'Two-thirds (4.3 in)': 4.3,
    'Half page (3.2 in)':  3.2,
}

# Point sizes at the chosen physical width — what Word will actually show.
FONT_BODY  = 8.0
FONT_LABEL = 8.5
FONT_TITLE = 9.5

PANELS = {
    'Both panels (price + SoC)': 'both',
    'Price only':                'price',
    'State of charge only':      'soc',
}

FORMATS = {
    'PNG (300 dpi)': ('png', 'image/png'),
    'SVG (vector)':  ('svg', 'image/svg+xml'),
    'PDF (vector)':  ('pdf', 'application/pdf'),
}

# Figure height as a fraction of the figure width, per panel type. Text does
# not shrink with the figure, so a narrow column needs proportionally more
# height for the axes to keep any room at all.
_ASPECT = {
    'price': {'wide': 0.34, 'narrow': 0.50},
    'soc':   {'wide': 0.50, 'narrow': 0.80},
    'both':  {'wide': 0.74, 'narrow': 1.15},
}

# Below this width, labels are shortened and the layout switches to 'narrow'.
_NARROW_IN = 5.0

_GRID  = dict(color='#d4d4d8', linewidth=0.4, alpha=0.9)
_PRICE = '#475569'


def _rc():
    """rcParams for a print figure. Sizes are points, so they hold in Word."""
    return {
        'font.size':          FONT_BODY,
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Calibri', 'Carlito', 'Arial', 'DejaVu Sans'],
        'axes.titlesize':     FONT_LABEL,
        'axes.labelsize':     FONT_LABEL,
        'xtick.labelsize':    FONT_BODY,
        'ytick.labelsize':    FONT_BODY,
        'legend.fontsize':    FONT_BODY,
        'figure.titlesize':   FONT_TITLE,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.major.size':   2.5,
        'ytick.major.size':   2.5,
        'axes.edgecolor':     '#52525b',
        'axes.facecolor':     'white',
        'figure.facecolor':   'white',
        'savefig.facecolor':  'white',
        'legend.frameon':     False,
        'svg.fonttype':       'none',   # keep SVG text as text for Word
        'pdf.fonttype':       42,
    }


def _date_locator(ax, width_in):
    """
    Tick density set by the physical width, not the span. A date label needs
    roughly 0.55 in at 8 pt, so a half-page figure gets correspondingly fewer
    ticks instead of a smear of overlapping text.
    """
    maxticks = max(3, int(width_in / 0.55))
    loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=True))


def _plot_price(ax, dt, prices, ccy, span_days):
    # Hairline over a full year, slightly heavier once zoomed in.
    lw = 0.35 if span_days > 120 else (0.6 if span_days > 21 else 0.9)
    ax.plot(dt, prices, color=_PRICE, linewidth=lw, solid_joinstyle='round')
    if len(prices) and np.nanmin(prices) < 0:
        ax.axhline(0, color='#a1a1aa', linewidth=0.5, linestyle=(0, (2, 2)))
    ax.set_ylabel('Price (' + ccy + '/MWh)')
    ax.grid(True, axis='y', **_GRID)
    ax.set_axisbelow(True)


def _plot_soc(ax, results, ccy, n_years, span_days, t0, t1,
              axis_to_capacity=False, width_in=6.5):
    lw       = 0.5 if span_days > 120 else (0.8 if span_days > 21 else 1.1)
    max_dur  = max(r['duration'] for r in results)
    peak_soc = 0.0

    compact = width_in < _NARROW_IN

    for i, r in enumerate(results):
        colour = PALETTE[i % len(PALETTE)][0]
        dt     = pd.to_datetime(r['dt_fine'])
        mask   = (dt >= t0) & (dt <= t1)
        soc    = np.asarray(r['soc_fine'])[mask]
        rev_k  = r['revenue'] / n_years / 1000
        rte    = int(round(r['rte'] * 100))
        # A narrow column cannot carry the full caption without the legend
        # overrunning the axes, so drop to the identifying part.
        label  = (f'{r["duration"]}h {r["tech_name"]} ({rte}%)' if compact else
                  f'{r["duration"]}h {r["tech_name"]} '
                  f'({rte}% RTE, {ccy}{rev_k:.0f}k/MW/yr)')
        ax.plot(dt[mask], soc, color=colour, linewidth=lw, label=label)
        ax.fill_between(dt[mask], 0, soc, color=colour, alpha=0.10, linewidth=0)
        if soc.size:
            peak_soc = max(peak_soc, float(soc.max()))

    # A long-duration technology rarely fills up, and scaling to its nameplate
    # flattens the shorter ones into an unreadable band along the axis. Default
    # to the range the window actually uses.
    top = max_dur if axis_to_capacity else max(peak_soc, 1e-6)

    # The legend sits inside the axes (an anchored one outside is not seen by
    # constrained layout and clips the y-label), so reserve headroom for it.
    ncol     = 2 if (width_in >= 6.0 and len(results) > 2) else 1
    rows     = -(-len(results) // ncol)
    headroom = 0.06 + 0.085 * rows
    top      = top * 1.02 / (1.0 - headroom)

    for i, r in enumerate(results):
        if r['duration'] <= top:     # only annotate a ceiling that is in view
            ax.axhline(r['duration'], color=PALETTE[i % len(PALETTE)][0],
                       linewidth=0.5, linestyle=(0, (3, 2)), alpha=0.7)

    ax.set_ylabel('SoC (h at 1 MW)' if compact else 'State of charge (h at 1 MW)')
    ax.set_ylim(-0.02 * top, top)
    ax.grid(True, axis='y', **_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', ncol=ncol, handlelength=1.5,
              borderaxespad=0.35, labelspacing=0.25, columnspacing=1.0,
              handletextpad=0.5)


def render_time_charts(datetimes, prices, results, *, country, year_label,
                       ccy='EUR', n_years=1, t0=None, t1=None,
                       panel='both', width_in=6.5, fmt='png', dpi=300,
                       title=True, axis_to_capacity=False):
    """
    Render the price and/or SoC panel as a Word-ready image.

    datetimes / prices : hourly series behind the price panel
    results            : per-technology dicts from build_chart (soc_fine, dt_fine, ...)
    t0, t1             : inclusive datetime window; defaults to the full series
    panel              : 'both' | 'price' | 'soc'
    width_in           : finished image width in inches, matched to the Word column
    fmt                : 'png' | 'svg' | 'pdf'
    axis_to_capacity   : scale the SoC axis to the largest nameplate rather than
                         to the state of charge actually reached in the window

    Returns the encoded image as bytes.
    """
    dt = pd.to_datetime(datetimes)
    t0 = pd.Timestamp(t0) if t0 is not None else dt[0]
    t1 = pd.Timestamp(t1) if t1 is not None else dt[-1]
    if t1 <= t0:
        raise ValueError('End of the export window must be after the start.')

    mask      = (dt >= t0) & (dt <= t1)
    span_days = max((t1 - t0).total_seconds() / 86400.0, 1e-6)
    height_in = width_in * _ASPECT[panel]['narrow' if width_in < _NARROW_IN else 'wide']

    with plt.rc_context(_rc()):
        # Constrained layout fits the content inside the requested figsize, so
        # the saved image is exactly width_in wide — no cropping, no rescaling.
        if panel == 'both':
            fig, axes = plt.subplots(
                2, 1, figsize=(width_in, height_in), sharex=True,
                height_ratios=[0.38, 0.62], layout='constrained',
            )
            fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.02)
            _plot_price(axes[0], dt[mask], np.asarray(prices)[mask], ccy, span_days)
            _plot_soc(axes[1], results, ccy, n_years, span_days, t0, t1,
                      axis_to_capacity, width_in)
            ax_bottom = axes[1]
        elif panel == 'price':
            fig, ax = plt.subplots(figsize=(width_in, height_in), layout='constrained')
            _plot_price(ax, dt[mask], np.asarray(prices)[mask], ccy, span_days)
            ax_bottom = ax
        else:
            fig, ax = plt.subplots(figsize=(width_in, height_in), layout='constrained')
            _plot_soc(ax, results, ccy, n_years, span_days, t0, t1,
                      axis_to_capacity, width_in)
            ax_bottom = ax

        ax_bottom.set_xlim(t0, t1)
        _date_locator(ax_bottom, width_in)
        for lbl in ax_bottom.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_horizontalalignment('center')

        if title:
            what = {'both':  'Price and state of charge',
                    'price': 'Day-ahead price',
                    'soc':   'State of charge'}[panel]
            fig.suptitle(f'{what} — {country} {year_label}', fontsize=FONT_TITLE)

        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi)
        plt.close(fig)

    return buf.getvalue()
