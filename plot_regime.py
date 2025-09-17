import os
import random
import argparse
from typing import List, Dict, Optional, Tuple
from collections import Counter
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_DATA_DIR = "data"
DEFAULT_DATA_FILE = "BTCUSDT_merged_2020-2025_labeled.csv"   # change when calling the script
DATE_COL_CANDIDATES = ["timestamp", "date", "datetime"]
PRICE_PREFER = ["close_5m", "close_1m", "close_15m", "mean_5m", "close"]  # preference order
REGIME_CANDIDATES = ["regime", "state", "regime_smooth", "state_smooth"]

CHUNK_COUNT = 6
CHUNK_LENGTH = 60
RANDOM_SEED = 42
MAJORITY_THRESHOLD = 0.5  # fraction required to call a dominant regime; otherwise "Mixed"

OUT_COMBINED = "regime_chunks_combined.png"
OUT_SEPARATE_PATTERN = "regime_chunk_{:02d}.png"
# -------------------------------------------------

# Optional human-readable labels for 5 regimes; fallback to generic "Regime X"
REGIME_LABELS: Dict[int, str] = {
    0: "Bearish Momentum",
    1: "Bearish Drift",
    2: "Neutral / Range",
    3: "Bullish Drift",
    4: "Bullish Momentum"
}


def find_date_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any datetime-like column
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def find_regime_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    # fallback: integer/low-unique numeric column
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            nuniq = df[c].nunique(dropna=True)
            if 2 <= nuniq <= 10:
                return c
    raise KeyError(f"None of the regime column candidates found: {candidates}")


def choose_price_column(df: pd.DataFrame, prefer_list: List[str]) -> str:
    for p in prefer_list:
        if p in df.columns:
            return p
    # fallback heuristics
    candidates = [c for c in df.columns if any(k in c for k in ("close", "mean", "price"))]
    if candidates:
        return candidates[0]
    raise KeyError("No suitable price column found. Provide one of: " + ", ".join(prefer_list))


def choose_chunk_starts_nonoverlap(n_rows: int, chunk_len: int, k: int, seed: int) -> List[int]:
    """
    Prefer non-overlapping chunks. If not enough room, will sample allowing overlap.
    Returns sorted list of start indices.
    """
    max_start = n_rows - chunk_len
    if max_start < 0:
        raise ValueError("Not enough rows for requested chunk length.")

    random.seed(seed)
    # possible non-overlapping starts at step = chunk_len
    candidate_starts = list(range(0, max_start + 1))
    random.shuffle(candidate_starts)

    starts = []
    for s in candidate_starts:
        # ensure non-overlap with existing starts
        if all(abs(s - prev) >= chunk_len for prev in starts):
            starts.append(s)
        if len(starts) >= k:
            break

    if len(starts) < k:
        # fallback to allowing overlaps
        extra = []
        attempts = 0
        while len(starts) + len(extra) < k and attempts < 10000:
            s = random.randint(0, max_start)
            if s not in starts and s not in extra:
                extra.append(s)
            attempts += 1
        starts += extra

    starts = sorted(starts)[:k]
    return starts


def prepare_data_single(path: str, date_col_candidates=DATE_COL_CANDIDATES) -> Tuple[pd.DataFrame, str, str]:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    df = pd.read_csv(path)
    # find date column
    date_col = find_date_column(df, date_col_candidates)
    if date_col is None:
        raise SystemExit("No date/timestamp column found. Expected one of: " + ", ".join(date_col_candidates))
    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        # attempt again with utc=False if any NaT present
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.sort_values(date_col).reset_index(drop=True)

    # regime column detection
    regime_col = find_regime_column(df, REGIME_CANDIDATES)
    print("Detected regime column:", regime_col)

    # price column selection
    price_col = choose_price_column(df, PRICE_PREFER)
    print("Using price column:", price_col)

    # ensure regime is integer-coded; if not, map to integers preserving order of appearance
    if not pd.api.types.is_integer_dtype(df[regime_col]):
        unique_vals = list(pd.Series(df[regime_col].dropna().unique()))
        unique_vals_sorted = unique_vals  # keep appearance order (not sorted) for reproducibility
        mapping = {v: i for i, v in enumerate(unique_vals_sorted)}
        df[regime_col] = df[regime_col].map(mapping)
        print(f"Mapped regime values to integer codes (mapping size = {len(mapping)}). Example mapping: {dict(list(mapping.items())[:5])}")
    df[regime_col] = df[regime_col].astype(int)

    return df, price_col, regime_col


def chunk_majority_label(chunk: pd.DataFrame, regime_col: str, threshold: float = MAJORITY_THRESHOLD) -> Tuple[Optional[int], bool]:
    """
    Returns (label, is_majority). If no majority >= threshold, returns (mode_label, False).
    """
    vals = chunk[regime_col].dropna().values
    if len(vals) == 0:
        return None, False
    cnt = Counter(vals)
    mode, count = cnt.most_common(1)[0]
    is_major = (count / len(vals)) >= threshold
    return int(mode), is_major


def build_color_map(unique_regimes: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {r: cmap(i % cmap.N) for i, r in enumerate(unique_regimes)}


def plot_chunks_grid(df: pd.DataFrame,
                     date_col: str,
                     price_col: str,
                     regime_col: str,
                     starts: List[int],
                     chunk_len: int,
                     out_combined: str,
                     out_sep_pattern: str,
                     majority_threshold: float = MAJORITY_THRESHOLD,
                     labels_map: Dict[int, str] = REGIME_LABELS):
    unique_regimes = sorted(df[regime_col].unique())
    color_map = build_color_map(unique_regimes)

    # legend handles
    legend_handles = [mpatches.Patch(color=color_map[r], label=labels_map.get(r, f"Regime {r}")) for r in unique_regimes]

    k = len(starts)
    # grid layout: choose ncols up to 6 for readability
    ncols = min(6, k)
    nrows = math.ceil(k / ncols)
    fig_w = max(12, 3 * ncols)
    fig_h = max(3 * nrows, 6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=False)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    points_summary = []

    for idx, (start_idx, ax) in enumerate(zip(starts, axes)):
        chunk = df.iloc[start_idx : start_idx + chunk_len]
        if chunk.empty:
            ax.set_visible(False)
            continue

        # majority label
        mode_label, is_major = chunk_majority_label(chunk, regime_col, majority_threshold)
        display_label = ("Mixed" if not is_major else labels_map.get(mode_label, f"Regime {mode_label}")) if mode_label is not None else "Unknown"

        # plot line + colored scatter by regime (per row)
        # convert regime -> color per row
        row_colors = [color_map.get(int(r), (0.7, 0.7, 0.7, 1.0)) for r in chunk[regime_col].values]
        ax.plot(chunk[date_col], chunk[price_col], color='gray', linewidth=0.7, alpha=0.9, zorder=1)
        ax.scatter(chunk[date_col], chunk[price_col], c=row_colors, s=30, edgecolors='none', zorder=2)
        ax.set_title(f"Chunk {idx+1}  [{start_idx}:{start_idx+chunk_len-1}] — {display_label}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.tick_params(axis='x', rotation=25)

        # store summary: mean price x, mean price y? here store mean price as y and mode as label
        mean_price = float(chunk[price_col].mean())
        points_summary.append({"chunk": idx+1, "start": start_idx, "label": mode_label, "is_major": is_major, "mean_price": mean_price})

    # hide extra axes if any
    for ax in axes[len(starts):]:
        ax.set_visible(False)

    # add legend on the right
    fig.legend(handles=legend_handles, title="Market Regimes", bbox_to_anchor=(0.95, 0.5), loc="center left")
    plt.tight_layout(rect=[0, 0, 0.93, 1.0])
    fig.savefig(out_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined grid plot:", out_combined)

    # also save separate images per chunk
    for i, start_idx in enumerate(starts, start=1):
        chunk = df.iloc[start_idx : start_idx + chunk_len]
        if chunk.empty:
            continue
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        row_colors = [color_map.get(int(r), (0.7, 0.7, 0.7, 1.0)) for r in chunk[regime_col].values]
        ax2.plot(chunk[date_col], chunk[price_col], color='gray', linewidth=0.7, alpha=0.9, zorder=1)
        ax2.scatter(chunk[date_col], chunk[price_col], c=row_colors, s=45, edgecolors='none', zorder=2)
        mode_label, is_major = chunk_majority_label(chunk, regime_col, majority_threshold)
        display_label = ("Mixed" if not is_major else labels_map.get(mode_label, f"Regime {mode_label}")) if mode_label is not None else "Unknown"
        ax2.set_title(f"Chunk {i} — Rows {start_idx} to {start_idx + chunk_len - 1} — {display_label}")
        ax2.set_ylabel("Price")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax2.tick_params(axis='x', rotation=25)
        fig2.legend(handles=legend_handles, title="Market Regimes", bbox_to_anchor=(1.02, 0.5), loc="center left")
        out_sep = out_sep_pattern.format(i)
        plt.tight_layout(rect=[0, 0, 0.88, 1.0])
        fig2.savefig(out_sep, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print("Saved separate plot:", out_sep)


def main(args):
    random.seed(args.random_seed)
    path = os.path.join(args.data_dir, args.data_file)
    df, price_col, regime_col = prepare_data_single(path)

    # choose starts
    starts = choose_chunk_starts_nonoverlap(len(df), args.chunk_length, args.chunk_count, args.random_seed)
    if len(starts) < args.chunk_count:
        print(f"Warning: only selected {len(starts)} chunks (requested {args.chunk_count}). Overlap allowed.")
    print("Chunk start indices:", starts)

    # plotting
    plot_chunks_grid(df,
                     date_col=find_date_column(df, DATE_COL_CANDIDATES),
                     price_col=price_col,
                     regime_col=regime_col,
                     starts=starts,
                     chunk_len=args.chunk_length,
                     out_combined=args.out_combined,
                     out_sep_pattern=args.out_separate_pattern,
                     majority_threshold=args.majority_threshold,
                     labels_map=REGIME_LABELS)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot regime-colored price chunks from a single labeled CSV.")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing CSV")
    p.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE, help="CSV filename")
    p.add_argument("--chunk-count", type=int, default=CHUNK_COUNT, help="Number of chunks to sample")
    p.add_argument("--chunk-length", type=int, default=CHUNK_LENGTH, help="Rows per chunk (contiguous)")
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed")
    p.add_argument("--majority-threshold", type=float, default=MAJORITY_THRESHOLD, help="Fraction required to call a majority regime (0-1)")
    p.add_argument("--out-combined", type=str, default=OUT_COMBINED, help="Output path for combined grid image")
    p.add_argument("--out-separate-pattern", type=str, default=OUT_SEPARATE_PATTERN, help="Pattern for separate images (format accepts index)")
    args = p.parse_args()
    main(args)
