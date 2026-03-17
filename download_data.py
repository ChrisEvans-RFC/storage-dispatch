#!/usr/bin/env python3
"""
download_data.py
================
Download hourly day-ahead / spot electricity prices for:

  • ERCOT  (Texas)               → ercot_wholesale_electricity_price_data_hourly/
  • CAISO  (California NP15/SP15) → caiso_wholesale_electricity_price_data_hourly/
  • AEMO   (Australia NEM)        → aemo_wholesale_electricity_price_data_hourly/

CSV format matches the European dataset:
  Country, ISO3 Code, Datetime (UTC), Datetime (Local), Price (<ccy>/MWh)
  — US files use USD/MWh; Australian files use AUD/MWh.

Requirements:
  pip install gridstatus nemosis pyarrow

Usage:
  python download_data.py                          # all markets, 2019–2024
  python download_data.py --market ercot
  python download_data.py --market aemo --start 2020 --end 2023
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ── ERCOT ─────────────────────────────────────────────────────────────────────

ERCOT_HUBS = {
    'ERCOT HB_NORTH':  'HB_NORTH',
    'ERCOT HB_SOUTH':  'HB_SOUTH',
    'ERCOT HB_WEST':   'HB_WEST',
    'ERCOT HB_HOUSTON': 'HB_HOUSTON',
}

def download_ercot(start_year: int, end_year: int, out_dir: str):
    try:
        import gridstatus
    except ImportError:
        sys.exit('gridstatus not installed. Run: pip install gridstatus')

    iso = gridstatus.Ercot()
    os.makedirs(out_dir, exist_ok=True)

    for display_name, hub_id in ERCOT_HUBS.items():
        rows = []
        for year in range(start_year, end_year + 1):
            print(f'  ERCOT {hub_id} {year}…', end=' ', flush=True)
            try:
                df = iso.get_lmp(
                    date=f'{year}-01-01',
                    end=f'{year + 1}-01-01',
                    market=gridstatus.Markets.DAY_AHEAD_HOURLY,
                    location_type='Hub',
                    verbose=False,
                )
                df = df[df['Location'] == hub_id].copy()
                if df.empty:
                    print(f'no data for {hub_id}')
                    continue
                df = df[['Time', 'LMP']].copy()
                df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)
                df.columns = ['Datetime (UTC)', 'Price (USD/MWh)']
                rows.append(df)
                print(f'{len(df):,} rows')
            except Exception as exc:
                print(f'ERROR: {exc}')

        if not rows:
            print(f'  → No data collected for {display_name}')
            continue

        result = pd.concat(rows).sort_values('Datetime (UTC)').drop_duplicates('Datetime (UTC)').reset_index(drop=True)
        result.insert(0, 'Country', display_name)
        result.insert(1, 'ISO3 Code', 'USA')
        result['Datetime (Local)'] = result['Datetime (UTC)'] - pd.Timedelta(hours=6)  # CST

        out_path = os.path.join(out_dir, f'{display_name}.csv')
        result[['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (USD/MWh)']].to_csv(out_path, index=False)
        print(f'  → Saved {len(result):,} rows to {out_path}')


# ── CAISO ─────────────────────────────────────────────────────────────────────

CAISO_HUBS = {
    'CAISO NP15': 'TH_NP15_GEN-APND',
    'CAISO SP15': 'TH_SP15_GEN-APND',
}

def download_caiso(start_year: int, end_year: int, out_dir: str):
    try:
        import gridstatus
    except ImportError:
        sys.exit('gridstatus not installed. Run: pip install gridstatus')

    iso = gridstatus.CAISO()
    os.makedirs(out_dir, exist_ok=True)

    for display_name, node_id in CAISO_HUBS.items():
        rows = []
        for year in range(start_year, end_year + 1):
            print(f'  CAISO {node_id} {year}…', end=' ', flush=True)
            try:
                df = iso.get_lmp(
                    date=f'{year}-01-01',
                    end=f'{year + 1}-01-01',
                    market=gridstatus.Markets.DAY_AHEAD_HOURLY,
                    location_type='Trading Hub',
                    verbose=False,
                )
                # gridstatus may return the short name or full node ID
                mask = df['Location'].isin([node_id, node_id.split('-')[0]])
                df = df[mask].copy()
                if df.empty:
                    # fall back: print available locations on first failure
                    avail = iso.get_lmp(
                        date=f'{year}-01-01',
                        end=f'{year}-01-02',
                        market=gridstatus.Markets.DAY_AHEAD_HOURLY,
                        location_type='Trading Hub',
                        verbose=False,
                    )['Location'].unique()
                    print(f'no match for {node_id}. Available: {list(avail)}')
                    continue
                df = df[['Time', 'LMP']].copy()
                df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)
                df.columns = ['Datetime (UTC)', 'Price (USD/MWh)']
                rows.append(df)
                print(f'{len(df):,} rows')
            except Exception as exc:
                print(f'ERROR: {exc}')

        if not rows:
            print(f'  → No data collected for {display_name}')
            continue

        result = pd.concat(rows).sort_values('Datetime (UTC)').drop_duplicates('Datetime (UTC)').reset_index(drop=True)
        result.insert(0, 'Country', display_name)
        result.insert(1, 'ISO3 Code', 'USA')
        result['Datetime (Local)'] = result['Datetime (UTC)'] - pd.Timedelta(hours=8)  # PST

        out_path = os.path.join(out_dir, f'{display_name}.csv')
        result[['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (USD/MWh)']].to_csv(out_path, index=False)
        print(f'  → Saved {len(result):,} rows to {out_path}')


# ── AEMO ──────────────────────────────────────────────────────────────────────

AEMO_REGIONS = {
    'New South Wales': 'NSW1',
    'Queensland':      'QLD1',
    'Victoria':        'VIC1',
    'South Australia': 'SA1',
}

def download_aemo(start_year: int, end_year: int, out_dir: str):
    try:
        from nemosis import dynamic_data_compiler
    except ImportError:
        sys.exit('nemosis not installed. Run: pip install nemosis')

    cache_dir = os.path.join(out_dir, '_nemosis_cache')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for region_name, region_id in AEMO_REGIONS.items():
        print(f'  AEMO {region_name} ({region_id}) {start_year}–{end_year}…', flush=True)
        try:
            df = dynamic_data_compiler(
                start_time=f'{start_year}/01/01 00:05:00',
                end_time=f'{end_year + 1}/01/01 00:00:00',
                table_name='DISPATCHPRICE',
                raw_data_cache=cache_dir,
                filter_cols=['REGIONID'],
                filter_values=[[region_id]],
                keep_csv=False,
            )
            # SETTLEMENTDATE is in NEM time = UTC+10 (AEST, no DST adjustment)
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
            df['Datetime (UTC)'] = df['SETTLEMENTDATE'] - pd.Timedelta(hours=10)

            # Resample 5-min intervals → hourly mean
            df = (df.set_index('Datetime (UTC)')[['RRP']]
                    .sort_index()
                    ['RRP']
                    .resample('h').mean()
                    .reset_index())
            df.columns = ['Datetime (UTC)', 'Price (AUD/MWh)']

            # Remove extreme negative spikes (floor at -1000 AUD/MWh for sensibility)
            pct_clipped = (df['Price (AUD/MWh)'] < -1000).mean() * 100
            if pct_clipped > 0:
                print(f'    Clipping {pct_clipped:.2f}% of hours below -1000 AUD/MWh')
                df['Price (AUD/MWh)'] = df['Price (AUD/MWh)'].clip(lower=-1000)

            df.insert(0, 'Country', region_name)
            df.insert(1, 'ISO3 Code', 'AUS')
            df['Datetime (Local)'] = df['Datetime (UTC)'] + pd.Timedelta(hours=10)

            out_path = os.path.join(out_dir, f'{region_name}.csv')
            df[['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (AUD/MWh)']].to_csv(out_path, index=False)
            print(f'  → Saved {len(df):,} rows to {out_path}')

        except Exception as exc:
            print(f'  ERROR for {region_id}: {exc}')
            raise


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download electricity price data')
    parser.add_argument('--start',  type=int, default=2019, help='First year (inclusive)')
    parser.add_argument('--end',    type=int, default=2024, help='Last year (inclusive)')
    parser.add_argument('--market', choices=['all', 'ercot', 'caiso', 'aemo'], default='all',
                        help='Which market(s) to download')
    args = parser.parse_args()

    if args.market in ('all', 'ercot'):
        print(f'\nDownloading ERCOT data {args.start}–{args.end}…')
        download_ercot(args.start, args.end, 'ercot_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'caiso'):
        print(f'\nDownloading CAISO data {args.start}–{args.end}…')
        download_caiso(args.start, args.end, 'caiso_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'aemo'):
        print(f'\nDownloading AEMO data {args.start}–{args.end}…')
        download_aemo(args.start, args.end, 'aemo_wholesale_electricity_price_data_hourly')

    print('\nDone!')


if __name__ == '__main__':
    main()
