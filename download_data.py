#!/usr/bin/env python3
"""
download_data.py
================
Download hourly day-ahead / spot electricity prices for:

  - CAISO  (California NP15/SP15)  -> caiso_wholesale_electricity_price_data_hourly/
  - ERCOT  (Texas hubs)            -> ercot_wholesale_electricity_price_data_hourly/
  - AEMO   (Australia NEM regions) -> aemo_wholesale_electricity_price_data_hourly/

CSV format matches the European dataset:
  Country, ISO3 Code, Datetime (UTC), Datetime (Local), Price (<ccy>/MWh)

Requirements:
  pip install nemosis pyarrow          # AEMO only
  (CAISO and ERCOT use only requests + pandas, already in requirements.txt)

Usage:
  python download_data.py                           # all markets, 2019-2024
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


def _biweekly_ranges(start_year: int, end_year: int):
    """Yield (start_date, end_date) pairs in 14-day chunks."""
    d = date(start_year, 1, 1)
    end = date(end_year + 1, 1, 1)
    while d < end:
        next_d = min(d + timedelta(days=14), end)
        yield d, next_d
        d = next_d


def _save(df: pd.DataFrame, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print(f'  -> Saved {len(df):,} rows to {out_path}')


# ── CAISO (OASIS API — no auth required) ─────────────────────────────────────
#
# API docs: https://www.caiso.com/documents/oasisapispecification.pdf
# Returns a ZIP containing a CSV with hourly DAM LMPs.
# NOTE: OASIS retains ~2 years of history. Data before ~2023 returns "No data".

CAISO_NODES = {
    'CAISO NP15': 'TH_NP15_GEN-APND',
    'CAISO SP15': 'TH_SP15_GEN-APND',
}
CAISO_URL = 'https://oasis.caiso.com/oasisapi/SingleZip'
CAISO_EARLIEST_YEAR = 2023  # OASIS only retains ~2 years; skip older requests


def _caiso_fetch_chunk(node_id: str, start: date, end: date,
                       session: requests.Session) -> pd.DataFrame | None:
    # CAISO operating day starts at Pacific midnight = T08:00 UTC (PST).
    # Using T08:00 is safe for year-level chunks; DST duplicates dropped later.
    params = {
        'queryname':     'PRC_LMP',
        'market_run_id': 'DAM',
        'node':          node_id,
        'startdatetime': start.strftime('%Y%m%dT08:00-0000'),
        'enddatetime':   end.strftime('%Y%m%dT08:00-0000'),
        'version':       '1',
        'resultformat':  '6',   # CSV inside a ZIP
    }
    for attempt in range(5):
        try:
            r = session.get(CAISO_URL, params=params, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_name = next((n for n in zf.namelist() if n.endswith('.csv')), None)
                if csv_name is None:
                    xml_name = next((n for n in zf.namelist() if n.endswith('.xml')), None)
                    if xml_name:
                        txt = zf.read(xml_name).decode('utf-8', errors='replace')
                        if 'No data returned' in txt or 'ERR_CODE' in txt:
                            return None   # outside retention window
                        print(f'    WARN: unexpected XML for {start}: {txt[:200]}')
                    return None
                df = pd.read_csv(zf.open(csv_name), low_memory=False)
                df = df[(df['LMP_TYPE'] == 'LMP') & (df['NODE'] == node_id)].copy()
                if df.empty:
                    return None
                df['Datetime (UTC)'] = pd.to_datetime(
                    df['INTERVALSTARTTIME_GMT'], utc=True).dt.tz_localize(None)
                return df[['Datetime (UTC)', 'MW']].rename(columns={'MW': 'Price (USD/MWh)'})
        except (zipfile.BadZipFile, requests.HTTPError):
            wait = 30 * (attempt + 1)
            print(f'\n    rate-limited, waiting {wait}s...', end=' ', flush=True)
            time.sleep(wait)
        except Exception as exc:
            print(f'    WARN: {start}: {exc}')
            return None
    print(f'    WARN: giving up on {start} after retries')
    return None


def download_caiso(start_year: int, end_year: int, out_dir: str):
    effective_start = max(start_year, CAISO_EARLIEST_YEAR)
    if effective_start > end_year:
        print(f'  CAISO OASIS only retains data from {CAISO_EARLIEST_YEAR}; nothing to download.')
        return
    if start_year < CAISO_EARLIEST_YEAR:
        print(f'  NOTE: CAISO OASIS retains ~2 years of data. '
              f'Skipping {start_year}-{CAISO_EARLIEST_YEAR-1}, downloading {effective_start}-{end_year}.')

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (storage-dispatch-tool)'

    for display_name, node_id in CAISO_NODES.items():
        print(f'\n  {display_name} ({node_id})')
        chunks = []
        current_year = None
        for start, end in _biweekly_ranges(effective_start, end_year):
            if start.year != current_year:
                current_year = start.year
                print(f'    {current_year}:', end=' ', flush=True)
            chunk = _caiso_fetch_chunk(node_id, start, end, session)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                print('.', end='', flush=True)
            time.sleep(8)   # ~7 req/min — well within OASIS rate limit
        print()

        if not chunks:
            print(f'  -> No data collected for {display_name}')
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
        time.sleep(60)  # pause between nodes


# ── ERCOT (public historical archive — no auth required) ─────────────────────
#
# ERCOT publishes yearly zipped CSVs of DAM Settlement Point Prices (NP4-190-CD)
# at their public CDR archive. Each file covers one calendar year.
# URL: https://www.ercot.com/files/docs/YYYY/NP4-190-CDYYYYMMDDYYYYMMDD.zip
# Fallback: try the "Historical DAM Settlement Point Prices" bulk file.

ERCOT_HUBS = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']


def _ercot_year_urls(year: int):
    """Return candidate URLs for ERCOT DAM SPP annual archive files."""
    y = str(year)
    return [
        f'https://www.ercot.com/files/docs/{y}/NP4-190-CD_{y}0101_{y}1231.zip',
        f'https://www.ercot.com/files/docs/{y}/Historical_DAM_Load_Zone_and_Hub_Prices.zip',
        f'https://www.ercot.com/files/docs/{y}/Historical_DAM_Settlement_Point_Prices.zip',
        f'https://www.ercot.com/files/docs/{y}/da_spp_{y}.zip',
    ]


def _ercot_fetch_year(year: int, session: requests.Session) -> pd.DataFrame | None:
    """Try each candidate URL for a year; return raw DataFrame if found."""
    for url in _ercot_year_urls(year):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_name = next((n for n in zf.namelist()
                                 if n.endswith('.csv') and not n.startswith('__')), None)
                if csv_name is None:
                    continue
                df = pd.read_csv(zf.open(csv_name), low_memory=False)
                print(f'      fetched from {url.split("/")[-1]}', flush=True)
                return df
        except (zipfile.BadZipFile, requests.HTTPError, Exception):
            continue
    return None


def download_ercot(start_year: int, end_year: int, out_dir: str):
    """
    Download ERCOT historical DAM Settlement Point Prices from their public archive.

    If this returns no data (ERCOT periodically restructures their archive URLs),
    register for a free API key at https://developer.ercot.com and re-run with
    the --ercot-key argument (to be implemented).
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/zip, application/octet-stream, */*',
        'Referer': 'https://www.ercot.com/',
    })

    # Collect all years into per-hub DataFrames
    hub_chunks: dict[str, list] = {h: [] for h in ERCOT_HUBS}

    for year in range(start_year, end_year + 1):
        print(f'    {year}...', end=' ', flush=True)
        df = _ercot_fetch_year(year, session)
        if df is None:
            print('no data (URL not found)')
            continue

        # Normalise column names (ERCOT changes them between years)
        df.columns = [c.strip().upper() for c in df.columns]

        # Find the key columns by common name variants
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'settlement' in cl and 'point' in cl and 'price' in cl and 'type' not in cl:
                col_map['price'] = col
            elif 'delivery' in cl and 'date' in cl:
                col_map['date'] = col
            elif 'hour' in cl and 'end' in cl:
                col_map['hour'] = col
            elif col in ('SETTLEMENTPOINT', 'SETTLEMENT_POINT', 'SETTLEMENT POINT'):
                col_map['node'] = col

        if not all(k in col_map for k in ('price', 'date', 'hour', 'node')):
            print(f'WARN: unrecognised columns: {list(df.columns)[:10]}')
            continue

        # Build datetime: hourEnding is 1-24, so subtract 1 for 0-based hour
        df['Datetime (Local)'] = pd.to_datetime(
            df[col_map['date']].astype(str) + ' ' +
            (df[col_map['hour']].astype(float).astype(int) - 1).astype(str).str.zfill(2) + ':00',
            format='%Y-%m-%d %H:%M', errors='coerce'
        )
        df['Datetime (UTC)'] = df['Datetime (Local)'] + pd.Timedelta(hours=6)  # CST

        for hub in ERCOT_HUBS:
            mask = df[col_map['node']].astype(str).str.strip() == hub
            sub = df[mask][['Datetime (UTC)', 'Datetime (Local)', col_map['price']]].copy()
            sub.columns = ['Datetime (UTC)', 'Datetime (Local)', 'Price (USD/MWh)']
            if not sub.empty:
                hub_chunks[hub].append(sub)

        rows_found = sum(len(df[df[col_map['node']].astype(str).str.strip() == h])
                         for h in ERCOT_HUBS)
        print(f'{rows_found:,} rows')
        time.sleep(2)

    os.makedirs(out_dir, exist_ok=True)
    for hub in ERCOT_HUBS:
        display_name = f'ERCOT {hub}'
        if not hub_chunks[hub]:
            print(f'  -> No data collected for {display_name}')
            continue
        result = (pd.concat(hub_chunks[hub])
                    .dropna(subset=['Datetime (UTC)'])
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
        print(f'\n  AEMO {region_name} ({region_id}) {start_year}-{end_year}...')
        try:
            df = dynamic_data_compiler(
                start_time=f'{start_year}/01/01 00:05:00',
                end_time=f'{end_year + 1}/01/01 00:00:00',
                table_name='DISPATCHPRICE',
                raw_data_location=cache_dir,
                filter_cols=['REGIONID'],
                filter_values=[[region_id]],
                keep_csv=False,
            )
            # SETTLEMENTDATE is NEM time = UTC+10 (AEST, no DST)
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
            df['Datetime (UTC)'] = df['SETTLEMENTDATE'] - pd.Timedelta(hours=10)

            # Resample 5-min dispatch intervals -> hourly mean
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
        print(f'\nDownloading CAISO data {args.start}-{args.end}...')
        download_caiso(args.start, args.end, 'caiso_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'ercot'):
        print(f'\nDownloading ERCOT data {args.start}-{args.end}...')
        download_ercot(args.start, args.end, 'ercot_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'aemo'):
        print(f'\nDownloading AEMO data {args.start}-{args.end}...')
        download_aemo(args.start, args.end, 'aemo_wholesale_electricity_price_data_hourly')

    print('\nDone!')


if __name__ == '__main__':
    main()
