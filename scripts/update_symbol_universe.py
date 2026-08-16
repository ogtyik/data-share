#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import us_universe_utils as base

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
UA = "Mozilla/5.0 (X11; Linux x86_64) pullback-scan-github-action/1.0"
SMALLCAP_AVG_DOLLAR_VOLUME_30D_USD = 15_000_000
DEFAULT_SCAN_PERIOD = '2mo'
DEFAULT_BATCH = 80
DEFAULT_SHARD_COUNT = 4
MANUAL_EXCLUSION_FILENAME = 'exclude_symbols.txt'
MONTHLY_EXCLUSION_FILENAME = 'monthly_excluded_symbols.json'
REQUIRED_CACHE_FILES = (
    'nasdaqlisted.txt',
    'otherlisted.txt',
    'us_symbols.csv',
    'monthly_excluded_symbols.json',
    'monthly_excluded_symbols.csv',
    'monthly_excluded_symbols.txt',
    'manifest.json',
)

# 新增：市值与股价过滤阈值（可在此调整）
MIN_MARKET_CAP_USD = 3_000_000_000   # 30亿美元
MIN_PRICE_USD = 12.0                 # 12美元


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def default_config_dir(root: Path) -> Path:
    return root / 'config'


def ensure_manual_exclusion_file(root: Path) -> Path:
    config_dir = default_config_dir(root)
    config_dir.mkdir(parents=True, exist_ok=True)
    manual_exclude_path = config_dir / MANUAL_EXCLUSION_FILENAME
    if not manual_exclude_path.exists():
        manual_exclude_path.write_text('# One symbol per line, e.g. AAPL or BRK.B\n', encoding='utf-8')
    return manual_exclude_path


def load_manual_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip().upper()
        if not line or line.startswith('#'):
            continue
        out.add(line)
        out.add(line.replace('-', '.'))
        out.add(line.replace('.', '-'))
    return out


def cache_is_fresh(out_dir: Path, *, skip_if_fresh_days: float) -> tuple[bool, str]:
    if skip_if_fresh_days <= 0:
        return False, 'skip disabled'
    manifest_path = out_dir / 'manifest.json'
    if not manifest_path.exists():
        return False, 'manifest missing'
    missing = [name for name in REQUIRED_CACHE_FILES if not (out_dir / name).exists()]
    if missing:
        return False, f'missing required cache files: {", ".join(missing)}'
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        updated = payload.get('updated_at_utc', '')
        updated_dt = datetime.fromisoformat(str(updated).replace('Z', '+00:00'))
    except Exception as e:
        return False, f'invalid manifest timestamp: {e}'
    age = now_utc() - updated_dt.astimezone(timezone.utc)
    threshold = timedelta(days=skip_if_fresh_days)
    if age <= threshold:
        return True, f'cache age {age} <= {threshold}'
    return False, f'cache age {age} > {threshold}'


def build_universe_frame(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    """构建初始普通股宇宙（不含任何市值/价格/流动性过滤）"""
    nasdaq = base.parse_nasdaq_listed(nasdaq_text)
    other = base.parse_other_listed(other_text)
    uni = pd.concat([nasdaq, other], ignore_index=True)
    uni = uni.drop_duplicates(subset=['Symbol']).reset_index(drop=True)
    uni['keep'] = uni.apply(lambda r: base.is_regular_security(r['Symbol'], r['name'], bool(r['etf']), bool(r['test_issue'])), axis=1)
    uni = uni[uni['keep']].copy().reset_index(drop=True)
    uni['Symbol'] = uni['Symbol'].map(lambda x: str(x).upper())
    uni['yahoo_symbol'] = uni['Symbol'].map(lambda x: base.yahoo_symbol(str(x)))
    return uni[['Symbol', 'yahoo_symbol', 'name', 'etf', 'test_issue', 'source']]


def filter_by_market_cap_and_price_shard(df: pd.DataFrame, log_file: Path) -> pd.DataFrame:
    """
    在分片内过滤市值和股价，使用 yf.Tickers 批量获取信息。
    只保留 市值 ≥ MIN_MARKET_CAP_USD 且 股价 ≥ MIN_PRICE_USD 的股票。
    """
    if df.empty:
        return df
    symbols = df['Symbol'].tolist()
    yahoo_symbols = [base.yahoo_symbol(s) for s in symbols]
    keep_symbols = []
    batch_size = 50  # 每批请求数量，避免超时
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"[{now_utc().isoformat()}] Shard filter: checking {len(symbols)} symbols\n")
        for i in range(0, len(yahoo_symbols), batch_size):
            batch = yahoo_symbols[i:i+batch_size]
            try:
                tickers = yf.Tickers(' '.join(batch))
            except Exception as e:
                log.write(f"ERROR creating Tickers for batch: {e}\n")
                continue
            for ys in batch:
                orig_sym = df[df['yahoo_symbol'] == ys]['Symbol'].iloc[0] if len(df[df['yahoo_symbol'] == ys]) > 0 else ys
                try:
                    info = tickers.tickers[ys].info
                    market_cap = info.get('marketCap')
                    price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
                    if market_cap is not None and price is not None:
                        if market_cap >= MIN_MARKET_CAP_USD and price >= MIN_PRICE_USD:
                            keep_symbols.append(orig_sym)
                        else:
                            log.write(f"FILTER_OUT {orig_sym} | marketCap={market_cap} | price={price}\n")
                    else:
                        log.write(f"INFO_MISSING {orig_sym} | marketCap={market_cap} | price={price}\n")
                except Exception as e:
                    log.write(f"INFO_ERROR {orig_sym} | {e}\n")
            time.sleep(0.1)  # 避免限流
    return df[df['Symbol'].isin(keep_symbols)].copy()


def build_exclusion_rows(df: pd.DataFrame, *, stderr_path: str, period: str, batch: int, phase: str):
    """
    下载历史K线，计算30日平均成交额，生成流动性排除列表。
    注意：传入的 df 已通过市值/股价过滤。
    """
    symbols = df['Symbol'].astype(str).tolist()
    mapped = {base.yahoo_symbol(sym): sym for sym in symbols}
    yahoo_symbols = list(mapped.keys())
    bars, misses = base.download_bars(yahoo_symbols, period, stderr_path, batch=batch, phase=phase)
    meta_by_symbol = {str(row['Symbol']): row for row in df.to_dict('records')}
    rows: list[dict] = []
    smallcap_symbols: list[str] = []
    missing_symbols: list[str] = []

    for yahoo_sym in sorted(misses):
        symbol = mapped.get(yahoo_sym, yahoo_sym)
        meta = meta_by_symbol.get(symbol, {})
        missing_symbols.append(symbol)
        rows.append({
            'symbol': symbol,
            'yahoo_symbol': yahoo_sym,
            'name': meta.get('name', ''),
            'reason': 'download_miss_or_possibly_delisted_or_not_yet_listed',
            'avg_dollar_volume_30d_usd': None,
            'valid_days': 0,
        })

    for yahoo_sym, price_df in bars.items():
        symbol = mapped.get(yahoo_sym, yahoo_sym)
        meta = meta_by_symbol.get(symbol, {})
        x = price_df.dropna(subset=['Close', 'Volume']).reset_index(drop=True)
        if len(x) == 0:
            missing_symbols.append(symbol)
            rows.append({
                'symbol': symbol,
                'yahoo_symbol': yahoo_sym,
                'name': meta.get('name', ''),
                'reason': 'empty_bars_or_possibly_delisted_or_not_yet_listed',
                'avg_dollar_volume_30d_usd': None,
                'valid_days': 0,
            })
            continue
        avg_dollar_volume_30d = base.trailing_avg_dollar_volume(x, len(x) - 1, days=30)
        if avg_dollar_volume_30d is not None and avg_dollar_volume_30d < SMALLCAP_AVG_DOLLAR_VOLUME_30D_USD:
            smallcap_symbols.append(symbol)
            rows.append({
                'symbol': symbol,
                'yahoo_symbol': yahoo_sym,
                'name': meta.get('name', ''),
                'reason': 'avg_dollar_volume_30d_below_15m_usd',
                'avg_dollar_volume_30d_usd': round(float(avg_dollar_volume_30d), 2),
                'valid_days': int(len(x)),
            })

    generated_symbols = sorted(set(smallcap_symbols) | set(missing_symbols))
    rows.sort(key=lambda row: (row['reason'], row['symbol']))
    return rows, generated_symbols, sorted(set(smallcap_symbols)), sorted(set(missing_symbols))


def workspace_dir_from_arg(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    return ROOT / '.tmp' / 'universe_update'


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_shard_frames(df: pd.DataFrame, workspace_dir: Path, shard_count: int) -> list[dict]:
    """将 DataFrame 拆分为多个分片 CSV 文件（仅包含原始宇宙，无过滤）"""
    shard_count = max(1, int(shard_count))
    symbols = df['Symbol'].astype(str).tolist()
    shard_symbol_lists = base.split_into_shards(symbols, shard_count)
    by_symbol = {str(row['Symbol']): row for row in df.to_dict('records')}
    shards_dir = workspace_dir / 'shards'
    shards_dir.mkdir(parents=True, exist_ok=True)
    shards_meta = []
    for idx, shard_symbols in enumerate(shard_symbol_lists, start=1):
        rows = [by_symbol[sym] for sym in shard_symbols if sym in by_symbol]
        shard_df = pd.DataFrame(rows, columns=['Symbol', 'yahoo_symbol', 'name', 'etf', 'test_issue', 'source'])
        shard_path = shards_dir / f'shard_{idx:02d}.csv'
        shard_df.to_csv(shard_path, index=False, encoding='utf-8')
        shards_meta.append({
            'shard_index': idx,
            'symbol_count': int(len(shard_df)),
            'path': str(shard_path.relative_to(workspace_dir)),
        })
    return shards_meta


def run_prepare(args) -> dict:
    """
    Prepare 阶段现在只做三件事：
    1. 检查缓存新鲜度（跳过可复用）
    2. 从 Nasdaq 实时抓取并解析普通股
    3. 将全量普通股列表分片（不进行任何市值/股价/流动性过滤）
    """
    root = ROOT
    out_dir = root / 'data' / 'universe'
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = workspace_dir_from_arg(args.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ensure_manual_exclusion_file(root)

    # 缓存新鲜度检查
    if not args.force_refresh:
        fresh, reason = cache_is_fresh(out_dir, skip_if_fresh_days=args.skip_if_fresh_days)
        if fresh:
            payload = {
                'status': 'skipped_fresh_cache',
                'reason': reason,
                'skip_if_fresh_days': args.skip_if_fresh_days,
                'force_refresh': args.force_refresh,
                'out_dir': str(out_dir),
                'workspace_dir': str(workspace_dir),
                'matrix': [],
            }
            write_json(workspace_dir / 'prepare.json', payload)
            return payload

    # 抓取 Nasdaq 数据
    source_dir = workspace_dir / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    nasdaq_text = fetch_text(NASDAQ_LISTED_URL)
    other_text = fetch_text(OTHER_LISTED_URL)
    (source_dir / 'nasdaqlisted.txt').write_text(nasdaq_text, encoding='utf-8')
    (source_dir / 'otherlisted.txt').write_text(other_text, encoding='utf-8')

    # 构建初始宇宙（仅过滤非普通股，不含市值/股价/流动性）
    df = build_universe_frame(nasdaq_text, other_text)

    if args.max_symbols and args.max_symbols > 0:
        df = df.head(args.max_symbols).copy()

    # 保存初始宇宙
    us_symbols_csv = source_dir / 'us_symbols.csv'
    df.to_csv(us_symbols_csv, index=False, encoding='utf-8')

    # 生成分片（未经任何过滤）
    shards_meta = write_shard_frames(df, workspace_dir, args.shard_count)

    payload = {
        'status': 'prepared',
        'generated_at_utc': now_utc().isoformat(),
        'period': args.period,
        'batch': int(args.batch),
        'shard_count': int(args.shard_count),
        'symbols': int(len(df)),
        'skip_if_fresh_days': args.skip_if_fresh_days,
        'force_refresh': args.force_refresh,
        'max_symbols': int(args.max_symbols),
        'workspace_dir': str(workspace_dir),
        'source_files': {
            'nasdaqlisted.txt': str((source_dir / 'nasdaqlisted.txt').relative_to(workspace_dir)),
            'otherlisted.txt': str((source_dir / 'otherlisted.txt').relative_to(workspace_dir)),
            'us_symbols.csv': str(us_symbols_csv.relative_to(workspace_dir)),
        },
        'shards': shards_meta,
        'matrix': [{'shard_index': item['shard_index']} for item in shards_meta],
    }
    write_json(workspace_dir / 'prepare.json', payload)
    return payload


def run_shard(args) -> dict:
    """
    Shard 阶段在分片内并行执行：
    1. 市值过滤（≥30亿美元）
    2. 股价过滤（≥12美元）
    3. 流动性过滤（30日平均成交额≥1500万美元）
    """
    workspace_dir = workspace_dir_from_arg(args.workspace_dir)
    prepare_payload = json.loads((workspace_dir / 'prepare.json').read_text(encoding='utf-8'))
    shard_index = int(args.shard_index)
    shard_path = workspace_dir / 'shards' / f'shard_{shard_index:02d}.csv'
    if not shard_path.exists():
        raise FileNotFoundError(f'shard file not found: {shard_path}')

    # 读取本分片的原始股票列表
    df = pd.read_csv(shard_path)
    if df.empty:
        # 空分片直接返回空结果
        payload = {
            'status': 'completed',
            'generated_at_utc': now_utc().isoformat(),
            'shard_index': shard_index,
            'symbols': 0,
            'period': args.period or prepare_payload['period'],
            'batch': int(args.batch or prepare_payload['batch']),
            'generated_symbols': [],
            'smallcap_symbols': [],
            'missing_symbols': [],
            'rows': [],
            'stderr_file': '',
        }
        results_dir = workspace_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        write_json(results_dir / f'shard_{shard_index:02d}.json', payload)
        return payload

    # ---- 第一步：市值与股价过滤 ----
    filter_log = workspace_dir / f'filter_shard_{shard_index:02d}.log'
    df = filter_by_market_cap_and_price_shard(df, filter_log)
    kept_symbols = df['Symbol'].tolist() if not df.empty else []   # 新增

    if df.empty:
        # 该分片无股票通过市值/股价过滤，返回空结果，但也要记录 kept_symbols（空）
        payload = {
            'status': 'completed',
            'generated_at_utc': now_utc().isoformat(),
            'shard_index': shard_index,
            'symbols': 0,
            'period': args.period or prepare_payload['period'],
            'batch': int(args.batch or prepare_payload['batch']),
            'generated_symbols': [],
            'smallcap_symbols': [],
            'missing_symbols': [],
            'rows': [],
            'kept_symbols': kept_symbols,      # 新增
            'stderr_file': str(filter_log.relative_to(workspace_dir)),
        }
        results_dir = workspace_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        write_json(results_dir / f'shard_{shard_index:02d}.json', payload)
        return payload

    # ---- 第二步：流动性过滤（下载K线，计算30日成交额） ----
    results_dir = workspace_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = results_dir / f'shard_{shard_index:02d}.stderr.log'
    if stderr_path.exists():
        stderr_path.unlink()

    rows, generated_symbols, smallcap_symbols, missing_symbols = build_exclusion_rows(
        df,
        stderr_path=str(stderr_path),
        period=args.period or prepare_payload['period'],
        batch=int(args.batch or prepare_payload['batch']),
        phase=f'MONTHLY_EXCLUSION_SHARD_{shard_index:02d}',
    )

    payload = {
        'status': 'completed',
        'generated_at_utc': now_utc().isoformat(),
        'shard_index': shard_index,
        'symbols': int(len(df)),
        'period': args.period or prepare_payload['period'],
        'batch': int(args.batch or prepare_payload['batch']),
        'generated_symbols': generated_symbols,
        'smallcap_symbols': smallcap_symbols,
        'missing_symbols': missing_symbols,
        'rows': rows,
        'kept_symbols': kept_symbols,      # 新增
        'stderr_file': str(stderr_path.relative_to(workspace_dir)),
        'filter_log': str(filter_log.relative_to(workspace_dir)),
    }
    write_json(results_dir / f'shard_{shard_index:02d}.json', payload)
    pd.DataFrame(rows).to_csv(results_dir / f'shard_{shard_index:02d}.csv', index=False, encoding='utf-8')
    return payload


def build_manifest(*, out_dir: Path, nasdaq_text: str, other_text: str, us_symbols_csv: Path, exclusion_json_path: Path, exclusion_csv_path: Path, exclusion_txt_path: Path, generated_symbols: list[str], smallcap_symbols: list[str], missing_symbols: list[str], updater_details: dict) -> dict:
    return {
        'updated_at_utc': now_utc().isoformat(),
        'sources': {
            'nasdaqlisted': NASDAQ_LISTED_URL,
            'otherlisted': OTHER_LISTED_URL,
            'yahoo_finance_daily_bars': 'yfinance',
        },
        'rules': {
            'universe_filter': 'Nasdaq Trader regular securities + Yahoo-friendly filter',
            'pre_scan_generated_exclusions': '30日平均成交額 < 1500萬美元，或 Yahoo 對不到 / 可能已退市 / 未上市代號',
            'market_cap_threshold_usd': MIN_MARKET_CAP_USD,
            'price_threshold_usd': MIN_PRICE_USD,
        },
        'counts': {
            'symbols': int(pd.read_csv(us_symbols_csv).shape[0]),
            'generated_exclusions': int(len(generated_symbols)),
            'smallcap_symbols': int(len(smallcap_symbols)),
            'missing_symbols': int(len(missing_symbols)),
        },
        'updater': updater_details,
        'files': {
            'nasdaqlisted.txt': {
                'sha256': sha256_bytes(nasdaq_text.encode('utf-8')),
                'bytes': len(nasdaq_text.encode('utf-8')),
            },
            'otherlisted.txt': {
                'sha256': sha256_bytes(other_text.encode('utf-8')),
                'bytes': len(other_text.encode('utf-8')),
            },
            'us_symbols.csv': {
                'sha256': sha256_bytes(us_symbols_csv.read_bytes()),
                'bytes': us_symbols_csv.stat().st_size,
            },
            'monthly_excluded_symbols.json': {
                'sha256': sha256_bytes(exclusion_json_path.read_bytes()),
                'bytes': exclusion_json_path.stat().st_size,
            },
            'monthly_excluded_symbols.csv': {
                'sha256': sha256_bytes(exclusion_csv_path.read_bytes()),
                'bytes': exclusion_csv_path.stat().st_size,
            },
            'monthly_excluded_symbols.txt': {
                'sha256': sha256_bytes(exclusion_txt_path.read_bytes()),
                'bytes': exclusion_txt_path.stat().st_size,
            },
        },
    }


def run_aggregate(args) -> dict:
    """汇总所有分片结果，生成最终的排除列表和缓存文件"""
    root = ROOT
    out_dir = root / 'data' / 'universe'
    out_dir.mkdir(parents=True, exist_ok=True)
    manual_exclude_path = ensure_manual_exclusion_file(root)
    workspace_dir = workspace_dir_from_arg(args.workspace_dir)
    prepare_payload = json.loads((workspace_dir / 'prepare.json').read_text(encoding='utf-8'))
    results_dir = workspace_dir / 'results'
    shard_files = sorted(results_dir.glob('shard_*.json'))
    expected = int(prepare_payload.get('shard_count', 0))
    if expected and len(shard_files) != expected:
        raise RuntimeError(f'expected {expected} shard results, found {len(shard_files)}')

    rows: list[dict] = []
    generated_symbols: set[str] = set()
    smallcap_symbols: set[str] = set()
    missing_symbols: set[str] = set()
    all_kept_symbols: set[str] = set()      # 新增：收集所有通过市值/股价过滤的符号
    shards_summary = []
    for shard_file in shard_files:
        payload = json.loads(shard_file.read_text(encoding='utf-8'))
        rows.extend(payload.get('rows', []))
        generated_symbols.update(payload.get('generated_symbols', []) or [])
        smallcap_symbols.update(payload.get('smallcap_symbols', []) or [])
        missing_symbols.update(payload.get('missing_symbols', []) or [])
        kept = payload.get('kept_symbols', [])
        all_kept_symbols.update(kept)        # 收集
        shards_summary.append({
            'shard_index': int(payload.get('shard_index', 0)),
            'symbols': int(payload.get('symbols', 0)),
            'generated_symbols': int(len(payload.get('generated_symbols', []) or [])),
            'missing_symbols': int(len(payload.get('missing_symbols', []) or [])),
        })

    rows.sort(key=lambda row: (row.get('reason', ''), row.get('symbol', '')))
    generated_symbols_sorted = sorted(generated_symbols)
    smallcap_symbols_sorted = sorted(smallcap_symbols)
    missing_symbols_sorted = sorted(missing_symbols)

    source_dir = workspace_dir / 'source'
    nasdaq_text = (source_dir / 'nasdaqlisted.txt').read_text(encoding='utf-8')
    other_text = (source_dir / 'otherlisted.txt').read_text(encoding='utf-8')
    us_symbols_csv_workspace = source_dir / 'us_symbols.csv'

    # ---- 新增：使用 all_kept_symbols 过滤 us_symbols.csv ----
    initial_df = pd.read_csv(us_symbols_csv_workspace)
    filtered_df = initial_df[initial_df['Symbol'].isin(all_kept_symbols)]
    # 覆盖原文件
    filtered_df.to_csv(us_symbols_csv_workspace, index=False, encoding='utf-8')
    # ------------------------------------------------

    filtered_symbol_count = len(filtered_df)

    (out_dir / 'nasdaqlisted.txt').write_text(nasdaq_text, encoding='utf-8')
    (out_dir / 'otherlisted.txt').write_text(other_text, encoding='utf-8')
    us_symbols_csv = out_dir / 'us_symbols.csv'
    us_symbols_csv.write_bytes(us_symbols_csv_workspace.read_bytes())

    exclusion_payload = {
        'updated_at_utc': now_utc().isoformat(),
        'thresholds': {
            'smallcap_avg_dollar_volume_30d_usd': SMALLCAP_AVG_DOLLAR_VOLUME_30D_USD,
            'market_data_period': args.period or prepare_payload['period'],
            'market_data_batch': int(args.batch or prepare_payload['batch']),
            'min_market_cap_usd': MIN_MARKET_CAP_USD,
            'min_price_usd': MIN_PRICE_USD,
        },
        'counts': {
            'symbols': filtered_symbol_count,      # 修改
            'smallcap_symbols': int(len(smallcap_symbols_sorted)),
            'missing_symbols': int(len(missing_symbols_sorted)),
            'generated_symbols': int(len(generated_symbols_sorted)),
            'manual_exclusions': int(len(load_manual_exclusions(manual_exclude_path))),
        },
        'generated_symbols': generated_symbols_sorted,
        'smallcap_symbols': smallcap_symbols_sorted,
        'missing_symbols': missing_symbols_sorted,
        'rows': rows,
    }
    exclusion_json_path = out_dir / MONTHLY_EXCLUSION_FILENAME
    exclusion_json_path.write_text(json.dumps(exclusion_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    exclusion_csv_path = out_dir / 'monthly_excluded_symbols.csv'
    pd.DataFrame(rows).to_csv(exclusion_csv_path, index=False, encoding='utf-8')
    exclusion_txt_path = out_dir / 'monthly_excluded_symbols.txt'
    exclusion_txt_path.write_text('\n'.join(generated_symbols_sorted) + ('\n' if generated_symbols_sorted else ''), encoding='utf-8')

    updater_details = {
        'mode': 'matrix_prepare_shard_aggregate',
        'shard_count': int(prepare_payload.get('shard_count', len(shard_files))),
        'period': args.period or prepare_payload['period'],
        'batch': int(args.batch or prepare_payload['batch']),
        'shards': sorted(shards_summary, key=lambda x: x['shard_index']),
    }
    manifest = build_manifest(
        out_dir=out_dir,
        nasdaq_text=nasdaq_text,
        other_text=other_text,
        us_symbols_csv=us_symbols_csv,
        exclusion_json_path=exclusion_json_path,
        exclusion_csv_path=exclusion_csv_path,
        exclusion_txt_path=exclusion_txt_path,
        generated_symbols=generated_symbols_sorted,
        smallcap_symbols=smallcap_symbols_sorted,
        missing_symbols=missing_symbols_sorted,
        updater_details=updater_details,
    )
    write_json(out_dir / 'manifest.json', manifest)
    payload = {
        'status': 'aggregated',
        'manifest': manifest,
        'manual_exclude_path': str(manual_exclude_path),
        'workspace_dir': str(workspace_dir),
    }
    write_json(workspace_dir / 'aggregate.json', payload)
    return payload


def run_full(args) -> dict:
    prepare_payload = run_prepare(args)
    if prepare_payload.get('status') == 'skipped_fresh_cache':
        return prepare_payload
    for item in prepare_payload.get('matrix', []):
        shard_ns = argparse.Namespace(
            workspace_dir=args.workspace_dir,
            shard_index=int(item['shard_index']),
            period=args.period,
            batch=args.batch,
        )
        run_shard(shard_ns)
    aggregate_ns = argparse.Namespace(
        workspace_dir=args.workspace_dir,
        period=args.period,
        batch=args.batch,
    )
    return run_aggregate(aggregate_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description='Update US universe cache and monthly exclusion list.')
    parser.add_argument('--mode', choices=['full', 'prepare', 'shard', 'aggregate'], default='full')
    parser.add_argument('--workspace-dir', default='')
    parser.add_argument('--shard-count', type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--max-symbols', type=int, default=0, help='Optional cap for smoke tests.')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--period', default=DEFAULT_SCAN_PERIOD)
    parser.add_argument('--stderr-path', default='')
    parser.add_argument('--skip-if-fresh-days', type=float, default=0, help='Skip rebuild if manifest/cache files are newer than this many days.')
    parser.add_argument('--force-refresh', action='store_true', help='Ignore freshness guard and rebuild anyway.')
    args = parser.parse_args()

    if args.mode == 'prepare':
        payload = run_prepare(args)
    elif args.mode == 'shard':
        if args.shard_index <= 0:
            raise SystemExit('--shard-index is required for --mode shard')
        payload = run_shard(args)
    elif args.mode == 'aggregate':
        payload = run_aggregate(args)
    else:
        payload = run_full(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
