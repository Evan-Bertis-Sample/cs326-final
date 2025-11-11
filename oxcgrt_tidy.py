from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

EPOCH = pd.Timestamp("2020-01-01")

IDENTITY_COLS = [
    "CountryName","CountryCode",
    "RegionName","RegionCode",
    "CityName","CityCode",
    "Jurisdiction"
]

DATE_COL_PATTERN = re.compile(r"^\d{2}[A-Za-z]{3}\d{4}$")  # e.g. 01Jan2020


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make OxCGRT data tidy (one row per place per date).")
    p.add_argument("--data-dir", required=True, type=str,
                   help="Path to the dataset root (folder containing OxCGRT_compact_national_v1.csv and timeseries_indices/).")
    p.add_argument("--compact-file", default="OxCGRT_compact_national_v1.csv", type=str,
                   help="Name of the main compact CSV (default: OxCGRT_compact_national_v1.csv).")
    p.add_argument("--timeseries-dir", default="timeseries_indices", type=str,
                   help="Subdirectory containing OxCGRT_timeseries_*_v1.csv (default: timeseries_indices).")
    p.add_argument("--indices", default="ALL", type=str,
                   help="Comma-separated list of indices to include (uses the <quantity> part of filenames). "
                        "Example: StringencyIndex_Average,ContainmentHealthIndex_Average. "
                        "Use 'ALL' (default) to include all found.")
    p.add_argument("--merge", nargs=2, metavar=("what", "how"), default=("indices", "outer"),
                   help="Merge control. 'what' is fixed to 'indices'. 'how' in {'left','outer','inner'}. Default outer.")
    p.add_argument("--national-only", action="store_true",
                   help="If set, keep only national rows (no Region/City).")
    p.add_argument("--out", required=True, type=str,
                   help="Output CSV path. Use .gz with --gzip to compress.")
    p.add_argument("--gzip", action="store_true",
                   help="Compress output as gzip. (Also works if file name ends with .gz)")
    p.add_argument("--chunksize", type=int, default=0,
                   help="Optional chunksize for reading the compact CSV. 0 means read at once.")
    p.add_argument("--date-col", default="DateISO", type=str,
                   help="Name for the normalized date column (default: DateISO).")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce printing; keep progress bars.")
    p.add_argument("--sort", action="store_true",
                help="Sort output by Country/Region/City and date (progression order).")

    return p.parse_args()


def log(msg: str, quiet: bool = False):
    if not quiet:
        print(msg, flush=True)


def read_compact(compact_path: Path, date_col_name: str, chunksize: int = 0,
                 national_only: bool = False, quiet: bool = False) -> pd.DataFrame:
    log(f"Reading compact file: {compact_path}", quiet)
    read_opts = dict(dtype=None, low_memory=False)

    if chunksize and chunksize > 0:
        frames = []
        for chunk in tqdm(pd.read_csv(compact_path, chunksize=chunksize, **read_opts),
                          desc="Compact chunks", unit="chunk"):
            chunk = normalize_compact_chunk(chunk, date_col_name, national_only)
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(compact_path, **read_opts)
        df = normalize_compact_chunk(df, date_col_name, national_only)

    log(f"Compact rows after filter: {len(df):,}", quiet)
    return df


def normalize_compact_chunk(df: pd.DataFrame, date_col_name: str, national_only: bool) -> pd.DataFrame:
    if "Date" in df.columns:
        df[date_col_name] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("Expected 'Date' column in compact file.")

    if national_only:
        mask = df["RegionName"].isna() & df["RegionCode"].isna() & df["CityName"].isna() & df["CityCode"].isna()
        df = df.loc[mask].copy()

    # Add days since epoch
    df["DaysSince2020"] = (df[date_col_name] - EPOCH).dt.days

    # Ensure identity keys exist even if missing
    for c in IDENTITY_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def find_timeseries_files(timeseries_dir: Path, include: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    files = []
    for p in sorted(timeseries_dir.glob("OxCGRT_timeseries_*_v1.csv")):
        m = re.match(r"OxCGRT_timeseries_(.+?)_v1\.csv$", p.name)
        if not m:
            continue
        qty = m.group(1)
        files.append((qty, p))

    if include is not None:
        include_set = set(include)
        files = [(q, p) for (q, p) in files if q in include_set]

    return files


def melt_timeseries(path: Path, date_col_name: str, quantity_name: str) -> pd.DataFrame:
    # Many date columns, few rows — read whole file
    df = pd.read_csv(path, low_memory=False)
    # Identify date columns by pattern like 01Jan2020
    date_cols = [c for c in df.columns if DATE_COL_PATTERN.match(c)]
    if not date_cols:
        # Some variants may use different caps or formats — try a fallback to anything like ddMmmYYYY ignoring leading zeros
        raise ValueError(f"No date-like columns found in {path}")

    long_df = df.melt(id_vars=[c for c in df.columns if c not in date_cols],
                      value_vars=date_cols,
                      var_name="DateToken",
                      value_name=quantity_name)

    # Convert DateToken to datetime
    long_df[date_col_name] = pd.to_datetime(long_df["DateToken"], format="%d%b%Y", errors="coerce")
    long_df.drop(columns=["DateToken"], inplace=True)

    # Keep identity cols and date + the metric column
    keep_cols = set(IDENTITY_COLS + [date_col_name, quantity_name])
    for c in IDENTITY_COLS:
        if c not in long_df.columns:
            long_df[c] = pd.NA

    cols = [*IDENTITY_COLS, date_col_name, quantity_name]
    long_df = long_df[cols]

    return long_df


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    compact_path = (data_dir / args.compact_file).resolve()
    timeseries_dir = (data_dir / args.timeseries_dir).resolve()

    if not compact_path.exists():
        raise SystemExit(f"Compact file not found: {compact_path}")

    # Read compact
    compact_df = read_compact(
        compact_path,
        date_col_name=args.date_col,
        chunksize=args.chunksize,
        national_only=args.national_only,
        quiet=args.quiet
    )

    # Decide which indices to include
    include_list: Optional[List[str]]
    if args.indices.strip().upper() == "ALL":
        include_list = None
    else:
        include_list = [s.strip() for s in args.indices.split(",") if s.strip()]

    # Gather timeseries files
    if timeseries_dir.exists():
        ts_files = find_timeseries_files(timeseries_dir, include_list)
    else:
        ts_files = []

    if not ts_files and (include_list is not None):
        print("Warning: No matching time-series files found for requested indices.")

    # Melt each and merge
    # We'll start from compact_df and merge each timeseries as new columns (left/outer/inner depending on args).
    what, how = args.merge
    if what != "indices":
        raise SystemExit("Only 'indices' merge mode is supported (this flag exists for future use).")
    if how not in {"left", "outer", "inner"}:
        raise SystemExit("Merge how must be one of: left, outer, inner")

    # Build an index for fast merging
    # Using a tuple key (CountryCode, RegionCode, CityCode, Jurisdiction, DateISO)
    merge_keys = [*IDENTITY_COLS, args.date_col]

    # Ensure proper dtypes for join keys
    for k in IDENTITY_COLS:
        compact_df[k] = compact_df[k].astype("string").astype(object)  # avoid categorical surprises

    # Progress bar over timeseries files
    if ts_files:
        tqdm_bar = tqdm(ts_files, desc="Merging timeseries", unit="file")
        for qty, path in tqdm_bar:
            tqdm_bar.set_postfix_str(qty)
            ts_long = melt_timeseries(path, args.date_col, quantity_name=qty)
            # Align dtypes
            for k in IDENTITY_COLS:
                ts_long[k] = ts_long[k].astype("string").astype(object)

            compact_df = compact_df.merge(ts_long, on=merge_keys, how=how)
    else:
        if not args.quiet:
            print("No timeseries_indices directory or files found; proceeding with compact file only.")

    # Guarantee a stable column order: identity -> DateISO -> DaysSince2020 -> rest (sorted)
    base_cols = [*IDENTITY_COLS, args.date_col, "DaysSince2020"]
    rest_cols = [c for c in compact_df.columns if c not in base_cols]

    # Put a few high-signal columns earlier if they exist
    preferred = [
        "StringencyIndex_Average", "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average", "EconomicSupportIndex"
    ]
    preferred_existing = [c for c in preferred if c in rest_cols]
    remaining = [c for c in rest_cols if c not in preferred_existing]
    ordered_cols = base_cols + preferred_existing + remaining

    compact_df = compact_df[ordered_cols]

    if args.sort:
        log("Sorting output by region and date...", args.quiet)
        sort_keys = ["CountryCode", "RegionCode", "CityCode", "Jurisdiction", args.date_col]
        # Only keep keys that exist in the dataframe (defensive)
        sort_keys = [k for k in sort_keys if k in compact_df.columns]
        compact_df.sort_values(sort_keys, inplace=True, ignore_index=True)

    # Write output
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compress = "gzip" if (args.gzip or str(out_path).endswith(".gz")) else None
    log(f"Writing output to: {out_path} (compression={compress})", args.quiet)


    compact_df.to_csv(out_path, index=False, compression=compress)

    # Quick summary
    n_places = compact_df[IDENTITY_COLS].drop_duplicates().shape[0]
    n_countries = compact_df["CountryName"].drop_duplicates().shape[0]
    n_rows = len(compact_df)
    min_date = compact_df[args.date_col].min()
    max_date = compact_df[args.date_col].max()
    log(f"Done. Rows: {n_rows:,} | Unique Countries {n_countries:,} | Unique places: {n_places:,} | Dates: {min_date.date()} → {max_date.date()}", args.quiet)


if __name__ == "__main__":
    main()
