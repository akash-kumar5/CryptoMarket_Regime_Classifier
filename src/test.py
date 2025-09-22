# scripts/test_merge_local.py
import os
import pandas as pd
from compute_features import (
    parse_depth_snapshot_json, load_and_normalize_funding, load_and_normalize_oi,
    aggregate_aggtrades_to_5m, aggregate_depth_snapshot_to_5m, merge_all_sources_to_5m
)

DATA = "data"
agg_path = os.path.join(DATA, "BTCUSDT_aggtrades_5s_20250920_20250922.csv")
depth_path = os.path.join(DATA, "BTCUSDT_depth_snapshot.json")
fund_path = os.path.join(DATA, "BTCUSDT_funding_20250920_20250922.csv")  # adjust if different
oi_path = os.path.join(DATA, "BTCUSDT_open_interest.json")
kline_path = None
# find combined kline
for f in os.listdir(DATA):
    if "combined" in f and f.endswith(".csv"):
        kline_path = os.path.join(DATA, f)
        break
if not kline_path:
    print("No combined kline CSV found in data/")
    raise SystemExit(1)

print("Loading files...")
k = pd.read_csv(kline_path, parse_dates=['timestamp'])
agg = pd.read_csv(agg_path)
depth = parse_depth_snapshot_json(depth_path)
fund = load_and_normalize_funding(fund_path)
oi = load_and_normalize_oi(oi_path)

print("Aggregating aggtrades -> 5m")
agg5 = aggregate_aggtrades_to_5m(agg)
print("agg5 head:\n", agg5.head())

print("Aggregating depth -> 5m")
depth5 = aggregate_depth_snapshot_to_5m(depth)
print("depth5 head:\n", depth5.head())

print("Merging into kline")
merged = merge_all_sources_to_5m(kline_df=k, agg_df=agg, depth_snapshots_df=depth, funding_df=fund, oi_df=oi)
print("Merged tail:\n", merged.tail(5).to_dict(orient='records')[-1])
