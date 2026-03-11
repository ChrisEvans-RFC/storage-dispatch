"""
UK Capacity Market: Storage Revenue vs Duration
================================================

Plots historical and forward CM payments for storage assets by duration,
combining NESO derating factor projections with T-4 auction clearing prices.

Data sources:
  - NESO Electricity Capacity Report (ECR): forward derating factors (Scaled EFC)
  - NESO/EMR Delivery Body: historical auction results
  - Modo Energy, Montel, Ofgem, DESNZ: historical derating & price data

Usage:
  python cm_storage_revenue_vs_duration.py              # saves PNG
  python cm_storage_revenue_vs_duration.py --show        # display interactively
  python cm_storage_revenue_vs_duration.py --output foo.png

Key concepts:
  - "Auction year" = year the T-4 auction is held (sets price for delivery 4yrs later)
  - Derating factor = % of nameplate MW that counts as "firm" capacity
  - CM payment = clearing price (£/kW/yr) × derating factor
  - Scaled EFC methodology introduced for 2025 auction onwards
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# =============================================================================
# DATA
# =============================================================================

# Storage durations in hours (half-hour increments up to 11h)
DURATIONS = [
    0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
    5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11
]

# ---------------------------------------------------------------------------
# NESO forward derating factors by AUCTION YEAR (%, Scaled EFC methodology)
# Source: NESO ECR / storage derating consultation
# None = no data published for that duration/year combination
# ---------------------------------------------------------------------------
DERATING_FORWARD = {
    2025: [6.59, 13.05, 19.65, 26.11, 32.57, 39.16, 45.63, 51.96,
           57.77, 62.82, 67.73, 72.64, 77.55, 82.33, 87.11, 94.37,
           94.37, 94.37, 94.37, 94.37, None,  None],
    2026: [6.08, 12.02, 18.09, 24.17, 30.11, 36.19, 42.27, 48.20,
           54.28, 60.08, 65.73, 71.38, 77.04, 82.69, 88.21, 93.72,
           94.37, 94.37, 94.37, 94.37, None,  None],
    2027: [5.35, 10.71, 15.92, 21.28, 26.63, 31.99, 37.34, 42.70,
           48.05, 53.27, 58.62, 63.97, 69.33, 74.54, 79.90, 85.12,
           90.47, 94.37, 94.37, 94.37, None,  None],
    2028: [4.98, 9.96,  14.93, 19.91, 24.89, 29.87, 34.71, 39.69,
           44.67, 49.64, 54.62, 59.60, 64.58, 69.56, 74.53, 79.38,
           84.36, 89.33, 94.18, 94.37, None,  None],
    2029: [4.79, 9.71,  14.51, 19.30, 24.22, 29.01, 33.81, 38.73,
           43.52, 48.31, 53.11, 58.03, 62.82, 67.61, 72.53, 77.33,
           82.12, 86.91, 91.70, 94.37, 94.37, 94.37],
    2030: [4.63, 9.25,  13.88, 18.50, 23.13, 27.75, 32.26, 36.88,
           41.51, 46.13, 50.76, 55.38, 60.01, 64.64, 69.26, 73.89,
           78.51, 83.01, 87.64, 92.14, 94.37, 94.37],
    2031: [4.47, 9.07,  13.55, 18.15, 22.62, 27.22, 31.70, 36.17,
           40.77, 45.25, 49.84, 54.32, 58.79, 63.39, 67.87, 72.47,
           76.94, 81.42, 86.02, 90.49, 94.37, 94.37],
    2032: [4.46, 8.92,  13.38, 17.84, 22.30, 26.76, 31.21, 35.67,
           40.13, 44.72, 49.18, 53.63, 58.09, 62.55, 67.01, 71.47,
           75.93, 80.39, 84.85, 89.18, 93.64, 94.37],
}

# ---------------------------------------------------------------------------
# Historical T-4 clearing prices (£/kW/yr) by AUCTION YEAR
# Source: NESO auction results, Ofgem annual reports
# ---------------------------------------------------------------------------
T4_PRICE_HISTORY = {
    2014: 19.40,    # DY 2018/19
    2015: 18.00,    # DY 2019/20
    2016: 22.50,    # DY 2020/21
    2017:  8.40,    # DY 2021/22
    2019:  6.44,    # DY 2022/23 (T-3, post CM reinstatement)
    2020: 18.00,    # DY 2024/25 (2019 T-4 for DY 2023/24 was £15.97)
    2021: 30.59,    # DY 2025/26
    2022: 63.00,    # DY 2026/27
    2023: 65.00,    # DY 2027/28
    2025: 60.00,    # DY 2028/29 (March 2025 auction)
}

# ---------------------------------------------------------------------------
# Historical storage derating factors (%, approximate) by AUCTION YEAR
# Reconstructed from Modo Energy, NESO ECR briefing notes, energy-storage.news
# Pre-2017: no duration tiering (all storage ~96%)
# 2017+: tiered by duration; values here are for T-4 auctions
# ---------------------------------------------------------------------------
DERATING_HISTORY = {
    # 2016 auction (DY 2020/21) — early tiered period
    2016: {
        0.5: 15,   1: 30,   1.5: 43,  2: 55,   2.5: 63,  3: 70,
        3.5: 76,   4: 82,   4.5: 86,  5: 89,   5.5: 91,  6: 93,
        6.5: 94,   7: 94.37, 7.5: 94.37, 8: 94.37,
    },
    # 2020 auction (DY 2024/25) — mid-decline period
    2020: {
        0.5: 7.5,  1: 19,   1.5: 28,  2: 36,   2.5: 43,  3: 49,
        3.5: 55,   4: 60,   4.5: 66,  5: 71,   5.5: 76,  6: 81,
        6.5: 85,   7: 89,   7.5: 92,  8: 94.37, 8.5: 94.37, 9: 94.37,
    },
    # 2023 auction (DY 2027/28) — pre-Scaled-EFC low point for short duration
    2023: {
        0.5: 3.5,  1: 8,    1.5: 12,  2: 15,   2.5: 19,  3: 23,
        3.5: 27,   4: 31,   4.5: 36,  5: 41,   5.5: 47,  6: 53,
        6.5: 60,   7: 67,   7.5: 76,  8: 85,   8.5: 94.37, 9: 94.37,
        9.5: 94.37, 10: 94.37,
    },
}

# ---------------------------------------------------------------------------
# Forward T-4 price scenarios (£/kW/yr)
#
# 2025 auction: £60/kW (known)
# 2026+ auctions: not yet held — range reflects uncertainty
#
# LOW  £25/kW — significant correction; 2026 T-4 has 50.4GW prequalified
#               vs 39.1GW target (Montel). But decarbonisation mandates on
#               gas plants and new contract thresholds support a higher floor
#               than the historic lows of £6-8/kW.
# MID  £45/kW — moderate correction from oversupply signal, offset by
#               ongoing gas/nuclear retirements and electrification demand.
# HIGH £65/kW — prices stay near recent levels; prequalified capacity does
#               not fully enter auction (historically large BESS drop-off).
# ---------------------------------------------------------------------------
PRICE_LOW  = 25.0
PRICE_MID  = 45.0
PRICE_HIGH = 65.0

# Forward derating year used for the uncertainty band
BAND_DERATING_YEAR = 2030

# Historical auction years to plot (subset for clarity)
HIST_YEARS_TO_PLOT = [2016, 2020, 2023]


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

COLORS_HIST = {
    2016: '#E8C090',    # light warm
    2020: '#D4A574',    # warm tan
    2023: '#C06030',    # burnt orange
}

COLOR_KNOWN  = '#1155AA'    # 2025 auction (known)
COLOR_BAND   = '#2266AA'    # forward band
COLOR_RFC    = '#003366'    # RFC annotation
COLOR_LIION  = '#882222'    # Li-ion annotation
COLOR_RATIO  = '#555555'    # multiplier annotation


# =============================================================================
# PLOTTING
# =============================================================================

def plot_cm_storage_revenue(output_path='cm_storage_revenue_vs_duration.png',
                            show=False, dpi=200):
    """Generate the CM storage revenue vs duration chart."""

    fig, ax = plt.subplots(figsize=(14, 9))

    # --- Forward uncertainty band (using BAND_DERATING_YEAR derating) ---
    dr_band = DERATING_FORWARD[BAND_DERATING_YEAR]
    band_durs, band_lo, band_mid, band_hi = [], [], [], []
    for i, d in enumerate(DURATIONS):
        if dr_band[i] is not None:
            band_durs.append(d)
            band_lo.append((dr_band[i] / 100) * PRICE_LOW)
            band_mid.append((dr_band[i] / 100) * PRICE_MID)
            band_hi.append((dr_band[i] / 100) * PRICE_HIGH)

    ax.fill_between(band_durs, band_lo, band_hi, alpha=0.12, color=COLOR_BAND)
    ax.plot(band_durs, band_hi, color=COLOR_BAND, lw=1.2, ls=':', alpha=0.5)
    ax.plot(band_durs, band_lo, color=COLOR_BAND, lw=1.2, ls=':', alpha=0.5)
    ax.plot(band_durs, band_mid, color=COLOR_BAND, lw=2.5, ls='-',
            label=f'~{BAND_DERATING_YEAR} forecast @ £{PRICE_MID:.0f}/kW (mid)')

    # Band labels
    idx_8h = DURATIONS.index(8)
    ax.text(8.5, band_hi[idx_8h] + 1.5, f'High: £{PRICE_HIGH:.0f}/kW',
            fontsize=8, color=COLOR_BAND, alpha=0.7)
    ax.text(8.5, band_lo[idx_8h] - 2.5, f'Low: £{PRICE_LOW:.0f}/kW',
            fontsize=8, color=COLOR_BAND, alpha=0.7)

    # --- 2025 auction line (known price) ---
    dr_2025 = DERATING_FORWARD[2025]
    durs_25, pay_25 = [], []
    for i, d in enumerate(DURATIONS):
        if dr_2025[i] is not None:
            durs_25.append(d)
            pay_25.append((dr_2025[i] / 100) * T4_PRICE_HISTORY[2025])
    ax.plot(durs_25, pay_25, color=COLOR_KNOWN, lw=3, ls='-', marker='s',
            markersize=4, label='2025 auction (£60/kW — known)', zorder=5)

    # --- Historical lines ---
    for year in HIST_YEARS_TO_PLOT:
        dr_dict = DERATING_HISTORY[year]
        price = T4_PRICE_HISTORY[year]
        durs = sorted(dr_dict.keys())
        payments = [(dr_dict[d] / 100) * price for d in durs]
        ax.plot(durs, payments, color=COLORS_HIST[year], lw=2.5, ls='--',
                marker='o', markersize=3.5, alpha=0.85,
                label=f'{year} auction (£{price:.0f}/kW)')

    # --- Reference lines ---
    ax.axhline(y=0.91 * PRICE_HIGH, color='grey', ls='-.', alpha=0.2, lw=0.8)
    ax.axhline(y=0.91 * PRICE_LOW,  color='grey', ls='-.', alpha=0.2, lw=0.8)

    # --- Annotations ---
    rfc_lo  = 94.37 / 100 * PRICE_LOW
    rfc_mid = 94.37 / 100 * PRICE_MID
    rfc_hi  = 94.37 / 100 * PRICE_HIGH
    ax.annotate(f'RFC 12hr\n£{rfc_lo:.0f}–{rfc_hi:.0f}/kW',
                xy=(11, rfc_mid), xytext=(11.5, 32),
                fontsize=10, color=COLOR_RFC, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_RFC, lw=1.5),
                ha='left')

    # Li-ion 2hr (use band derating year)
    li_dr = DERATING_FORWARD[BAND_DERATING_YEAR][DURATIONS.index(2)]
    li_lo  = li_dr / 100 * PRICE_LOW
    li_mid = li_dr / 100 * PRICE_MID
    li_hi  = li_dr / 100 * PRICE_HIGH
    ax.annotate(f'Li-ion 2hr\n£{li_lo:.0f}–{li_hi:.0f}/kW',
                xy=(2, li_mid), xytext=(3.3, 5),
                fontsize=10, color=COLOR_LIION, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_LIION, lw=1.5),
                ha='left')

    # Multiplier arrow
    ratio = rfc_mid / li_mid
    ax.annotate('', xy=(0.8, rfc_mid), xytext=(0.8, li_mid),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RATIO, lw=1.5))
    ax.text(0.15, (rfc_mid + li_mid) / 2, f'~{ratio:.0f}×', fontsize=11,
            color=COLOR_RATIO, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLOR_RATIO, alpha=0.9))

    # --- Formatting ---
    ax.set_xlabel('Storage Duration (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Capacity Market Payment (£/kW/yr)', fontsize=13, fontweight='bold')
    ax.set_title(
        'UK Capacity Market: Storage Revenue vs Duration\n'
        f'Historical and Forward Range ({BAND_DERATING_YEAR} derating, '
        f'£{PRICE_LOW:.0f}–{PRICE_HIGH:.0f}/kW T-4 price band)',
        fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 70)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('£%.0f'))

    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9,
              title='Auction Year (T-4 Clearing Price)', title_fontsize=10)
    ax.grid(True, alpha=0.15)

    ax.text(0.98, 0.02,
            f'2025 auction: £60/kW (known). Forward band: £{PRICE_LOW:.0f}–{PRICE_HIGH:.0f}/kW range.\n'
            'Low scenario: 2026 auction oversupply (50.4GW prequalified vs 39.1GW target).\n'
            'High scenario: sustained gas exit + electrification demand pressure.\n'
            f'Forward derating: NESO ECR {BAND_DERATING_YEAR} projections. '
            'Historical derating approximate.\n'
            'Source: NESO, Modo Energy, Montel, Ofgem, DESNZ',
            transform=ax.transAxes, fontsize=7.5, color='grey',
            ha='right', va='bottom', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Chart saved to {output_path}")

    if show:
        plt.show()
    plt.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_cm_payment(duration_hours, derating_pct, t4_price):
    """Calculate CM payment in £/kW/yr."""
    return (derating_pct / 100) * t4_price


def get_derating(duration_hours, year, source='forward'):
    """
    Look up derating factor (%) for a given duration and year.

    Parameters
    ----------
    duration_hours : float
        Storage duration (must be in DURATIONS list, or 12 for extrapolation).
    year : int
        Auction year.
    source : str
        'forward' for NESO projections, 'history' for historical.

    Returns
    -------
    float or None
    """
    if duration_hours >= 11 and source == 'forward':
        # Beyond published data — use cap (same as CCGT technical availability)
        return 94.37

    if source == 'forward':
        if year not in DERATING_FORWARD:
            return None
        if duration_hours not in DURATIONS:
            return None
        idx = DURATIONS.index(duration_hours)
        return DERATING_FORWARD[year][idx]

    elif source == 'history':
        if year not in DERATING_HISTORY:
            return None
        return DERATING_HISTORY[year].get(duration_hours, None)


def print_summary_table():
    """Print a summary table of CM payments across scenarios."""
    dr_band = DERATING_FORWARD[BAND_DERATING_YEAR]

    print(f"\n{'='*65}")
    print(f"  CM Payment (£/kW/yr) — {BAND_DERATING_YEAR} derating factors")
    print(f"{'='*65}")
    print(f"{'Duration':>8}  {'Low £'+str(int(PRICE_LOW)):>10}  "
          f"{'Mid £'+str(int(PRICE_MID)):>10}  "
          f"{'High £'+str(int(PRICE_HIGH)):>10}  {'12h/Xh ratio':>12}")
    print(f"{'-'*65}")

    for d in [1, 2, 4, 6, 8, 10, 12]:
        if d in DURATIONS:
            idx = DURATIONS.index(d)
            dr = dr_band[idx] if dr_band[idx] is not None else 94.37
        elif d >= 11:
            dr = 94.37
        else:
            continue

        lo = dr / 100 * PRICE_LOW
        mi = dr / 100 * PRICE_MID
        hi = dr / 100 * PRICE_HIGH
        r  = 94.37 / dr if dr > 0 else 0
        print(f"{d:>6}hr  £{lo:>8.1f}  £{mi:>8.1f}  £{hi:>8.1f}  {r:>10.1f}×")

    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UK Capacity Market: Storage Revenue vs Duration chart')
    parser.add_argument('--output', '-o', default='cm_storage_revenue_vs_duration.png',
                        help='Output file path (default: cm_storage_revenue_vs_duration.png)')
    parser.add_argument('--show', action='store_true',
                        help='Display chart interactively')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output resolution (default: 200)')
    args = parser.parse_args()

    plot_cm_storage_revenue(output_path=args.output, show=args.show, dpi=args.dpi)
    print_summary_table()
