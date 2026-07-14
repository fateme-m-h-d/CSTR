import pandas as pd
from pathlib import Path
import argparse

def read_report_csv(path):
    df = pd.read_csv(path, header=None, names=["key", "value"])
    out = dict(zip(df["key"], df["value"]))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model", type=str, default="KKThPINN")
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--val_ratio", type=str, default="0.2")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    base = Path(f"./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}")

    rows = []
    for run in range(args.runs):
        path = base / f"{args.model_id}_{args.val_ratio}_{run}.csv"
        if not path.exists():
            print(f"Missing: {path}")
            continue

        d = read_report_csv(path)
        d["run"] = run
        rows.append(d)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "rmse_total",
        "rmse_inner",
        "violation",
        "violation_g1",
        "violation_g2",
        "cb1_cb2_l2",
        "ca_weight",
        "cb1_weight",
        "cb2_weight",
        "cc_weight",
        "violation_max_any_constraint",
        "violation_max_sample_mean",
        "violation_g1_max",
        "violation_g2_max",
        "worst_region_violation",
        "worst_region_id",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = df[numeric_cols].agg(["mean", "std", "min", "max"])

    out_csv = base / f"{args.model_id}_{args.val_ratio}_summary_{args.runs}runs.csv"
    summary.to_csv(out_csv)

    print("\nPer-run results:")
    print(df[["run", "rmse_total", "rmse_inner", "violation", "violation_g1",
    "violation_g2", "cb1_cb2_l2"]])

    print("\nSummary:")
    print(summary)

    print(f"\nSaved summary to: {out_csv}")

if __name__ == "__main__":
    main()