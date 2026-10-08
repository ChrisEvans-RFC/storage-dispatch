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

ERCOT credentials (free):
  ERCOT blocks unauthenticated downloads. Register at https://developer.ercot.com,
  subscribe to the "Public" product, create an app, then pass:
    --ercot-user EMAIL --ercot-pass PASS --ercot-client-id ID --ercot-sub-key KEY
  Or export as env vars: ERCOT_USER, ERCOT_PASS, ERCOT_CLIENT_ID, ERCOT_SUB_KEY

Usage:
  python download_data.py                           # all markets, 2019-2024
  python download_data.py --market caiso
  python download_data.py --market aemo --start 2020 --end 2023
  python download_data.py --market ercot --start 2022 --end 2024 \
    --ercot-user your@email.com --ercot-pass yourpass \
    --ercot-client-id CLIENT_ID --ercot-sub-key SUB_KEY
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


# ── ERCOT (developer API — free account required) ─────────────────────────────
#
# ERCOT requires a free developer account for all data access (even public reports).
# Direct file downloads from ercot.com/files/docs/ return 403.
#
# Setup (one-time, 5 minutes):
#   1. Register at https://developer.ercot.com
#   2. Subscribe to the "Public" product (free, under Products tab)
#   3. Copy your Primary Subscription Key from your Profile page
#   4. Create an Application (under My Apps) — note the Client ID shown
#
# Then run:
#   python download_data.py --market ercot --start 2022 --end 2024 \
#     --ercot-user your@email.com --ercot-pass yourpassword \
#     --ercot-client-id YOUR_CLIENT_ID --ercot-sub-key YOUR_SUB_KEY
#
# Alternatively set environment variables to keep credentials out of shell history:
#   ERCOT_USER, ERCOT_PASS, ERCOT_CLIENT_ID, ERCOT_SUB_KEY

ERCOT_HUBS      = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']
ERCOT_TOKEN_URL = ('https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/'
                   'B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token')
ERCOT_API_BASE  = 'https://api.ercot.com/api/public-reports'


def _ercot_get_token(username: str, password: str, client_id: str) -> str:
    r = requests.post(ERCOT_TOKEN_URL, data={
        'grant_type':    'password',
        'username':      username,
        'password':      password,
        'response_type': 'token',
        'scope':         f'openid {client_id}',
        'client_id':     client_id,
    }, timeout=30)
    r.raise_for_status()
    return r.json()['access_token']


def _ercot_fetch_page(token: str, sub_key: str,
                      date_from: str, date_to: str,
                      hub: str, page: int = 1, size: int = 1000) -> dict:
    r = requests.get(
        f'{ERCOT_API_BASE}/np4-190-cd/',
        headers={
            'Authorization':             f'Bearer {token}',
            'Ocp-Apim-Subscription-Key': sub_key,
        },
        params={
            'deliveryDateFrom': date_from,
            'deliveryDateTo':   date_to,
            'settlementPoint':  hub,
            'size':             size,
            'page':             page,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _ercot_parse_response(resp: dict) -> pd.DataFrame | None:
    """
    Parse ERCOT API v2 JSON into a flat DataFrame.
    Handles both dict-per-row and columns+rows-as-arrays response formats.
    """
    data = resp.get('data', [])
    if not data:
        return None
    if isinstance(data[0], dict):
        return pd.DataFrame(data)
    fields = [f['label'] if isinstance(f, dict) else str(f)
              for f in resp.get('fields', [])]
    return pd.DataFrame(data, columns=fields)


def download_ercot(start_year: int, end_year: int, out_dir: str,
                   username: str = '', password: str = '',
                   client_id: str = '', sub_key: str = ''):
    """
    Download ERCOT DAM Settlement Point Prices (NP4-190-CD) via the developer API.
    Credentials fall back to ERCOT_USER / ERCOT_PASS / ERCOT_CLIENT_ID / ERCOT_SUB_KEY
    environment variables if not passed directly.
    """
    username  = username  or os.environ.get('ERCOT_USER', '')
    password  = password  or os.environ.get('ERCOT_PASS', '')
    client_id = client_id or os.environ.get('ERCOT_CLIENT_ID', '')
    sub_key   = sub_key   or os.environ.get('ERCOT_SUB_KEY', '')

    if not all([username, password, client_id, sub_key]):
        print(
            '  ERCOT credentials not provided. Register for a free account at\n'
            '  https://developer.ercot.com, then run:\n'
            '\n'
            '    python download_data.py --market ercot \\\n'
            '      --ercot-user YOUR_EMAIL --ercot-pass YOUR_PASSWORD \\\n'
            '      --ercot-client-id YOUR_CLIENT_ID --ercot-sub-key YOUR_SUB_KEY\n'
            '\n'
            '  Or export env vars: ERCOT_USER, ERCOT_PASS, ERCOT_CLIENT_ID, ERCOT_SUB_KEY'
        )
        return

    print('  Obtaining ERCOT API token...', end=' ', flush=True)
    try:
        token = _ercot_get_token(username, password, client_id)
        print('OK')
    except Exception as exc:
        print(f'FAILED: {exc}')
        return

    hub_chunks: dict[str, list] = {h: [] for h in ERCOT_HUBS}

    for year in range(start_year, end_year + 1):
        date_from = f'{year}-01-01'
        date_to   = f'{year}-12-31'

        for hub in ERCOT_HUBS:
            print(f'    {year} {hub}...', end=' ', flush=True)
            page = 1
            total = 0

            while True:
                try:
                    resp = _ercot_fetch_page(token, sub_key, date_from, date_to,
                                             hub, page)
                except requests.HTTPError as exc:
                    if exc.response.status_code == 401:
                        token = _ercot_get_token(username, password, client_id)
                        resp  = _ercot_fetch_page(token, sub_key, date_from, date_to,
                                                  hub, page)
                    else:
                        print(f'HTTP {exc.response.status_code}')
                        break

                df_page = _ercot_parse_response(resp)
                if df_page is None or df_page.empty:
                    break

                hub_chunks[hub].append(df_page)
                total += len(df_page)

                meta        = resp.get('_meta', {})
                total_pages = meta.get('totalPages', page)
                if page >= total_pages:
                    break
                page += 1
                time.sleep(0.2)

            print(f'{total:,} records')

    os.makedirs(out_dir, exist_ok=True)
    for hub in ERCOT_HUBS:
        display_name = f'ERCOT {hub}'
        if not hub_chunks[hub]:
            print(f'  -> No data collected for {display_name}')
            continue

        df = pd.concat(hub_chunks[hub], ignore_index=True)

        # Map column names (ERCOT API uses camelCase)
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'price' in cl and 'settlement' in cl:
                col_map['price'] = col
            elif 'date' in cl and 'delivery' in cl:
                col_map['date'] = col
            elif 'hour' in cl and 'end' in cl:
                col_map['hour'] = col

        # hourEnding is "HH:00" or integer 1-24; convert to 0-based hour
        hour_raw = df[col_map.get('hour', 'hourEnding')].astype(str)
        hour_num = (pd.to_numeric(hour_raw.str.extract(r'(\d+)', expand=False),
                                  errors='coerce').astype('Int64') - 1)
        date_str = df[col_map.get('date', 'deliveryDate')].astype(str)

        df['Datetime (Local)'] = pd.to_datetime(
            date_str + ' ' + hour_num.astype(str).str.zfill(2) + ':00',
            format='%Y-%m-%d %H:%M', errors='coerce',
        )
        df['Datetime (UTC)'] = df['Datetime (Local)'] + pd.Timedelta(hours=6)  # CST

        price_col = col_map.get('price', 'settlementPointPrice')
        df['Price (USD/MWh)'] = pd.to_numeric(df[price_col], errors='coerce')

        result = (df[['Datetime (UTC)', 'Datetime (Local)', 'Price (USD/MWh)']]
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
    # ERCOT developer API credentials (free account at developer.ercot.com)
    parser.add_argument('--ercot-user',      default='',
                        help='ERCOT developer account email (or set ERCOT_USER env var)')
    parser.add_argument('--ercot-pass',      default='',
                        help='ERCOT developer account password (or set ERCOT_PASS env var)')
    parser.add_argument('--ercot-client-id', default='',
                        help='ERCOT API client ID from developer.ercot.com (or ERCOT_CLIENT_ID)')
    parser.add_argument('--ercot-sub-key',   default='',
                        help='ERCOT API subscription key (or set ERCOT_SUB_KEY env var)')
    args = parser.parse_args()

    if args.market in ('all', 'caiso'):
        print(f'\nDownloading CAISO data {args.start}-{args.end}...')
        download_caiso(args.start, args.end, 'caiso_wholesale_electricity_price_data_hourly')

    if args.market in ('all', 'ercot'):
        print(f'\nDownloading ERCOT data {args.start}-{args.end}...')
        download_ercot(args.start, args.end, 'ercot_wholesale_electricity_price_data_hourly',
                       username=args.ercot_user, password=args.ercot_pass,
                       client_id=args.ercot_client_id, sub_key=args.ercot_sub_key)

    if args.market in ('all', 'aemo'):
        print(f'\nDownloading AEMO data {args.start}-{args.end}...')
        download_aemo(args.start, args.end, 'aemo_wholesale_electricity_price_data_hourly')

    print('\nDone!')


if __name__ == '__main__':
    main()
