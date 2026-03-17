#!/usr/bin/env python3
"""
download_data.py
================
Download hourly day-ahead / spot electricity prices for:

  • CAISO  (California NP15/SP15)  → caiso_wholesale_electricity_price_data_hourly/
  • ERCOT  (Texas hubs)            → ercot_wholesale_electricity_price_data_hourly/
  • AEMO   (Australia NEM regions) → aemo_wholesale_electricity_price_data_hourly/

CSV format matches the European dataset:
  Country, ISO3 Code, Datetime (UTC), Datetime (Local), Price (<ccy>/MWh)

Requirements:
  pip install nemosis pyarrow          # AEMO only
  (CAISO and ERCOT use only requests + pandas, already in requirements.txt)

Usage:
  python download_data.py                           # all markets, 2019–2024
  python download_data.py --market caiso
  python download_data.py --market aemo --start 2020 --end 2023
  python download_data.py --market ercot --start 2022 --end 2024
"""

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests


# ── shared helpers ────────────────────────────────────────────────────────────

def _month_ranges(start_year: int, end_year: int):
    """Yield (start_date, end_date) pairs covering each calendar month."""
    d = date(start_year, 1, 1)
    end = date(end_year + 1, 1, 1)
    while d < end:
        next_d = (d + relativedelta(months=1))
        yield d, min(next_d, end)
        d = next_d


def _save(df: pd.DataFrame, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print(f'  → Saved {len(df):,} rows to {out_path}')


# ── CAISO (OASIS API — no auth required) ─────────────────────────────────────
#
# API docs: https://www.caiso.com/documents/oasisapispecification.pdf
# Returns a ZIP containing a CSV with hourly DAM LMPs.

CAISO_NODES = {
    'CAISO NP15': 'TH_NP15_GEN-APND',
    'CAISO SP15': 'TH_SP15_GEN-APND',
}
CAISO_URL = 'https://oasis.caiso.com/oasisapi/SingleZip'


def _caiso_fetch_month(node_id: str, start: date, end: date,
                       session: requests.Session) -> pd.DataFrame | None:
    params = {
        'queryname':     'PRC_LMP',
        'market_run_id': 'DAM',
        'node':          node_id,
        'startdatetime': start.strftime('%Y%m%dT00:00-0000'),
        'enddatetime':   end.strftime('%Y%m%dT00:00-0000'),
        'version':       '1',
        'resultformat':  '6',   # CSV inside a ZIP
    }
    for attempt in range(3):
        try:
            r = session.get(CAISO_URL, params=params, timeout=120)
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt == 2:
                print(f'    WARN: {exc}')
                return None
            time.sleep(5 * (attempt + 1))

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))
            df = pd.read_csv(zf.open(csv_name), low_memory=False)
    except Exception as exc:
        print(f'    WARN: could not parse ZIP for {start}: {exc}')
        return None

    # Keep only LMP values for the requested node
    df = df[(df['LMP_TYPE'] == 'LMP') & (df['NODE'] == node_id)].copy()
    if df.empty:
        return None

    df['Datetime (UTC)'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'], utc=True).dt.tz_localize(None)
    df = df[['Datetime (UTC)', 'MW']].rename(columns={'MW': 'Price (USD/MWh)'})
    return df


def download_caiso(start_year: int, end_year: int, out_dir: str):
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (storage-dispatch-tool)'

    for display_name, node_id in CAISO_NODES.items():
        print(f'\n  {display_name} ({node_id})')
        chunks = []
        for start, end in _month_ranges(start_year, end_year):
            print(f'    {start:%Y-%m}…', end=' ', flush=True)
            chunk = _caiso_fetch_month(node_id, start, end, session)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                print(f'{len(chunk):,} rows')
            else:
                print('no data')
            time.sleep(1)   # be polite to OASIS

        if not chunks:
            print(f'  → No data collected for {display_name}')
            continue

        result = (pd.concat(chunks)
                    .sort_values('Datetime (UTC)')
                    .drop_duplicates('Datetime (UTC)')
                    .reset_index(drop=True))
        result.insert(0, 'Country',   display_name)
        result.insert(1, 'ISO3 Code', 'USA')
        result['Datetime (Local)'] = result['Datetime (UTC)'] - pd.Timedelta(hours=8)
        _save(result[['Country', 'ISO3 Code', 'Datetime (UTC)',
                       'Datetime (Local)', 'Price (USD/MWh)']], out_dir, f'{display_name}.csv')


# ── ERCOT (public REST API — no auth required for public reports) ─────────────
#
# ERCOT public API: https://api.ercot.com/api/public-reports
# Report NP4-190-CD: DAM Settlement Point Prices (Load Zones & Hubs)

ERCOT_HUBS = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']
ERCOT_BASE = 'https://api.ercot.com/api/public-reports/np4-190-cd'


def _ercot_fetch_month(hub: str, start: date, end: date,
                       session: requests.Session) -> pd.DataFrame | None:
    """Fetch one month of ERCOT DAM SPP for a single hub via the public API."""
    url = f'{ERCOT_BASE}/lz_and_hub_da_spp'
    params = {
        'deliveryDateFrom': start.strftime('%Y-%m-%d'),
        'deliveryDateTo':   (end - timedelta(days=1)).strftime('%Y-%m-%d'),
        'settlementPoint':  hub,
        'size':             10000,
        'page':             1,
    }
    rows = []
    while True:
        for attempt in range(3):
            try:
                r = session.get(url, params=params, timeout=60)
                r.raise_for_status()
                break
            except Exception as exc:
                if attempt == 2:
                    print(f'    WARN: {exc}')
                    return pd.DataFrame(rows) if rows else None
                time.sleep(5)

        data = r.json()
        page_rows = data.get('data', [])
        rows.extend(page_rows)

        meta = data.get('_meta', {})
        if params['page'] >= meta.get('totalPages', 1):
            break
        params['page'] += 1

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Expected columns: deliveryDate, hourEnding, settlementPoint, settlementPointPrice
    df['hour'] = df['hourEnding'].astype(str).str.zfill(2)
    df['Datetime (Local)'] = pd.to_datetime(
        df['deliveryDate'].astype(str) + ' ' + (df['hourEnding'].astype(int) - 1).astype(str).str.zfill(2) + ':00',
        format='%Y-%m-%d %H:%M'
    )
    df['Datetime (UTC)']   = df['Datetime (Local)'] + pd.Timedelta(hours=6)  # CST = UTC-6
    df = df[['Datetime (UTC)', 'Datetime (Local)', 'settlementPointPrice']].copy()
    df.columns = ['Datetime (UTC)', 'Datetime (Local)', 'Price (USD/MWh)']
    return df


def download_ercot(start_year: int, end_year: int, out_dir: str):
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (storage-dispatch-tool)'

    for hub in ERCOT_HUBS:
        display_name = f'ERCOT {hub}'
        print(f'\n  {display_name}')
        chunks = []
        for start, end in _month_ranges(start_year, end_year):
            print(f'    {start:%Y-%m}…', end=' ', flush=True)
            chunk = _ercot_fetch_month(hub, start, end, session)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                print(f'{len(chunk):,} rows')
            else:
                print('no data')
            time.sleep(0.5)

        if not chunks:
            print(f'  → No data collected for {display_name}')
            continue

        result = (pd.concat(chunks)
                    .sort_values('Datetime (UTC)')
                    .drop_duplicates('Datetime (UTC)')
                    .reset_index(drop=True))
        result.insert(0, 'Country',   display_name)
        result.insert(1, 'ISO3 Code', 'USA')
        _save(result[['Country', 'ISO3 Code', 'Datetime (UTC)',
                       'Datetime (Local)', 'Price (USD/MWh)']], out_dir, f'{display_name}.csv')


# ── AEMO (nemosis — no auth required) ────────────────────────────────────────

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
        sys.exit('nemosis not installed. Run: pip install nemosis pyarrow')

    cache_dir = os.path.join(out_dir, '_nemosis_cache')
    os.makedirs(out_dir,    exist_ok=True)
    os.makedirs(cache_dir,  exist_ok=True)

    for region_name, region_id in AEMO_REGIONS.items():
        print(f'\n  AEMO {region_name} ({region_id}) {start_year}–{end_year}…')
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
            # SETTLEMENTDATE is NEM time = UTC+10 (AEST, no DST)
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
            df['Datetime (UTC)'] = df['SETTLEMENTDATE'] - pd.Timedelta(hours=10)

            # Resample 5-min dispatch intervals → hourly mean
            df = (df.set_index('Datetime (UTC)')[['RRP']]
                    .sort_index()['RRP']
                    .resample('h').mean()
                    .reset_index())
            df.columns = ['Datetime (UTC)', 'Price (AUD/MWh)']

            n_clipped = (df['Price (AUD/MWh)'] < -1000).sum()
            if n_clipped:
                print(f'    Clipping {n_clipped} hours below -1000 AUD/MWh')
                df['Price (AUD/MWh)'] = df['Price (AUD/MWh)'].clip(lower=-1000)

            df.insert(0, 'Country',   region_name)
            df.insert(1, 'ISO3 Code', 'AUS')
            df['Datetime (Local)'] = df['Datetime (UTC)'] + pd.Timedelta(hours=10)

            _save(df[['Country', 'ISO3 Code', 'Datetime (UTC)',
                       'Datetime (Local)', 'Price (AUD/MWh)']], out_dir, f'{region_name}.csv')

        except Exception as exc:
            print(f'  ERROR for {region_id}: {exc}')
            raise


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download electricity price data')
    parser.add_argument('--start',  type=int, default=2019, help='First year (inclusive)')
    parser.add_argument('--end',    type=int, default=2024, help='Last year (inclusive)')
    parser.add_argument('--market', choices=['all', 'ercot', 'caiso', 'aemo'], default='all')
    args = parser.parse_args()

    if args.market in ('all', 'caiso'):
        print(f'\nDownloading CAISO data {args.start}–{args.end}…')
        download_caiso(args.start, args.end, 'caiso_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'ercot'):
        print(f'\nDownloading ERCOT data {args.start}–{args.end}…')
        download_ercot(args.start, args.end, 'ercot_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'aemo'):
        print(f'\nDownloading AEMO data {args.start}–{args.end}…')
        download_aemo(args.start, args.end, 'aemo_wholesale_electricity_price_data_hourly')

    print('\nDone!')


if __name__ == '__main__':
    main()
