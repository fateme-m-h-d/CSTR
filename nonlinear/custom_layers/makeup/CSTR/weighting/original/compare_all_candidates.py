from pathlib import Path
import argparse
import pandas as pd


def get_value(df, stat, col):
    if stat in df.index and col in df.columns:
        return df.loc[stat, col]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="KKThPINN")
    parser.add_argument("--dataset_type", type=str, default="cstr")
    parser.add_argument("--val_ratio", type=str, default="0.2")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--prefix", type=str, default="top")  # use top or grid
    args = parser.parse_args()

    base = Path(f"./data/tables/{args.dataset_type}/{args.model}/{args.val_ratio}")

    pattern = f"{args.prefix}_*_{args.val_ratio}_summary_{args.runs}runs.csv"
    files = sorted(base.glob(pattern))

    if not files:
        print(f"No files found with pattern: {base / pattern}")
        return

    rows = []

    for path in files:
        df = pd.read_csv(path, index_col=0)

        model_id = path.name.replace(
            f"_{args.val_ratio}_summary_{args.runs}runs.csv", ""
        )

        row = {
            "model_id": model_id,

            "rmse_mean": get_value(df, "mean", "rmse_total"),
            "rmse_std": get_value(df, "std", "rmse_total"),

            "violation_mean": get_value(df, "mean", "violation"),
            "violation_std": get_value(df, "std", "violation"),

            "violation_g1_mean": get_value(df, "mean", "violation_g1"),
            "violation_g2_mean": get_value(df, "mean", "violation_g2"),

            "cb1_cb2_l2_mean": get_value(df, "mean", "cb1_cb2_l2"),

            "ca_weight": get_value(df, "mean", "ca_weight"),
            "cb1_weight": get_value(df, "mean", "cb1_weight"),
            "cb2_weight": get_value(df, "mean", "cb2_weight"),
            "cc_weight": get_value(df, "mean", "cc_weight"),
        }

        rows.append(row)

    results = pd.DataFrame(rows)

    results = results.sort_values("violation_mean")

    out_csv = base / f"{args.prefix}_candidates_ranked_{args.runs}runs.csv"
    results.to_csv(out_csv, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nAll candidate results ranked by violation:\n")
    print(results)

    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()