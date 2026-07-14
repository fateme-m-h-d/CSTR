from pathlib import Path
import pandas as pd

base = Path("./data/tables/cstr/KKThPINN/0.2")

rows = []

for path in base.glob("cbpen_lam*_0.2_summary_3runs.csv"):
    df = pd.read_csv(path, index_col=0)

    row = {
        "model_id": path.name.replace("_0.2_summary_3runs.csv", ""),
        "rmse_mean": df.loc["mean", "rmse_total"],
        "rmse_std": df.loc["std", "rmse_total"],
        "violation_mean": df.loc["mean", "violation"],
        "violation_std": df.loc["std", "violation"],
        "violation_g1_mean": df.loc["mean", "violation_g1"],
        "violation_g2_mean": df.loc["mean", "violation_g2"],
        "cb1_cb2_l2_mean": df.loc["mean", "cb1_cb2_l2"],
        "ca_weight": df.loc["mean", "ca_weight"],
        "cb1_weight": df.loc["mean", "cb1_weight"],
        "cb2_weight": df.loc["mean", "cb2_weight"],
        "cc_weight": df.loc["mean", "cc_weight"],
    }

    rows.append(row)

ranked = pd.DataFrame(rows)
ranked = ranked.sort_values("violation_mean")

out = base / "cbpen_ranked_by_violation_3runs.csv"
ranked.to_csv(out, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(ranked)
print(f"\nSaved: {out}")