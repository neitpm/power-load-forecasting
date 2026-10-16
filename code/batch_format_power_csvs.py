# batch_format_power_csvs.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NUM_COLS = ["UA", "UB", "UC", "IA", "IB", "IC", "CosF", "P_kW", "Q_kVAr"]


def _parse_time_to_datetime(df: pd.DataFrame, time_col: str = "TIME") -> pd.Series:
    """
    Robust datetime parser.
    - If TIME already includes date (e.g. 2024-01-01 00:00:00.000) -> parse directly
    - Else try combine with DATE/Ngay if available + TIME as timedelta
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing column '{time_col}'")

    s = df[time_col].astype(str).str.strip()

    # Try direct parse (works when TIME already has full datetime)
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    ok_ratio = dt.notna().mean()

    if ok_ratio >= 0.8:
        return dt

    # Fallback: combine DATE + TIME (timedelta)
    date_col = None
    for cand in ["DATE", "Date", "Ngay", "NGAY", "Ngày"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # No date info => return NaT (user must provide date)
        return pd.to_datetime(pd.Series([pd.NaT] * len(df)))

    d = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)

    # TIME like "30:00.0" (mm:ss.s) -> prepend "00:" -> "00:30:00.0"
    t = s
    # strip trailing .0/.000 etc is ok for to_timedelta
    t = t.str.replace(r"\s+", "", regex=True)
    t = t.where(t.str.count(":") == 2, "00:" + t)

    td = pd.to_timedelta(t, errors="coerce")
    return d + td


def clean_one_file(
    in_path: Path,
    out_dir: Path,
    freq: str = "30min",
    make_wide: bool = True,
    make_xlsx: bool = False,
    time_col: str = "TIME",
    point_col: str = "MaDiemDo",
    text_cols=("MBA", "TenDiemDo"),
    num_cols=DEFAULT_NUM_COLS,
) -> dict:
    df = pd.read_csv(in_path, low_memory=False)

    # Trim text columns
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Parse datetime
    df["TIME_dt"] = _parse_time_to_datetime(df, time_col=time_col)

    # Coerce numeric columns
    present_num = [c for c in num_cols if c in df.columns]
    for c in present_num:
        # handle comma decimal
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")

    # Standard TIME string (no milliseconds)
    df[time_col] = df["TIME_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Sort
    sort_cols = [c for c in ["TIME_dt", point_col] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- Export LONG ---
    out_long = out_dir / f"{in_path.stem}_clean_long.csv"
    base_cols = [c for c in ["MBA", "MAMBA", point_col, "TenDiemDo", time_col] if c in df.columns]
    df_long = df[base_cols + present_num].copy()
    df_long.to_csv(out_long, index=False, encoding="utf-8")

    out_wide = None

    # --- Export WIDE ---
    if make_wide:
        df_w = df.dropna(subset=["TIME_dt"]).copy()
        if point_col not in df_w.columns:
            raise ValueError(f"Missing column '{point_col}' for wide pivot")

        # pivot_table handles duplicates more safely than pivot
        wide = df_w.pivot_table(
            index="TIME_dt",
            columns=point_col,
            values=present_num,
            aggfunc="first",
        )
        # Flatten columns: VAR_<MaDiemDo>
        wide.columns = [f"{var}_{int(pt)}" for (var, pt) in wide.columns]
        wide = wide.sort_index()

        # Full 30-min timeline
        if len(wide.index) > 0:
            start = wide.index.min()
            end = wide.index.max()
            full_index = pd.date_range(start=start, end=end, freq=freq)
            wide = wide.reindex(full_index)

        wide = wide.reset_index().rename(columns={"index": "TIME_dt"})
        wide[time_col] = wide["TIME_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
        wide = wide[[time_col] + [c for c in wide.columns if c not in [time_col, "TIME_dt"]]]

        out_wide = out_dir / f"{in_path.stem}_wide_{freq}.csv"
        wide.to_csv(out_wide, index=False, encoding="utf-8")

    # --- Export XLSX (optional) ---
    out_xlsx = None
    if make_xlsx:
        out_xlsx = out_dir / f"{in_path.stem}_formatted.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_long_x = df_long.copy()
            df_long_x[time_col] = pd.to_datetime(df_long_x[time_col], errors="coerce")
            df_long_x.to_excel(writer, sheet_name="Long", index=False)

            if make_wide and out_wide is not None:
                wide_x = pd.read_csv(out_wide)
                wide_x[time_col] = pd.to_datetime(wide_x[time_col], errors="coerce")
                wide_x.to_excel(writer, sheet_name=f"Wide_{freq}", index=False)

    stats = {
        "file": str(in_path.name),
        "rows_long": int(df_long.shape[0]),
        "unique_points": int(df_long[point_col].nunique()) if point_col in df_long.columns else None,
        "time_min": str(df["TIME_dt"].min()),
        "time_max": str(df["TIME_dt"].max()),
        "out_long": str(out_long),
        "out_wide": str(out_wide) if out_wide else None,
        "out_xlsx": str(out_xlsx) if out_xlsx else None,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing CSV files")
    ap.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern, e.g. *.csv")
    ap.add_argument("--out_dir", type=str, default="formatted_out", help="Output folder")
    ap.add_argument("--freq", type=str, default="30min", help="Timeline frequency for wide file")
    ap.add_argument("--wide", type=int, default=1, help="1=export wide, 0=skip")
    ap.add_argument("--xlsx", type=int, default=0, help="1=export xlsx, 0=skip")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found: {in_dir} / {args.pattern}")

    all_stats = []
    for p in files:
        try:
            st = clean_one_file(
                p,
                out_dir,
                freq=args.freq,
                make_wide=bool(args.wide),
                make_xlsx=bool(args.xlsx),
            )
            all_stats.append(st)
            print(f"{p.name} -> ok")
        except Exception as e:
            print(f"{p.name} -> {e}")

    # Save summary
    summary_path = out_dir / "_summary.csv"
    pd.DataFrame(all_stats).to_csv(summary_path, index=False, encoding="utf-8")
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
