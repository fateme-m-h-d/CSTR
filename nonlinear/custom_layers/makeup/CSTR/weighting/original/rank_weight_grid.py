from pathlib import Path
import pandas as pd

base = Path("./data/tables/cstr/KKThPINN/0.2")

rows = []

for path in base.glob("grid_*_0.2_summary_3runs.csv"):
    df = pd.read_csv(path, index_col=0)

    row = {
        "model_id": path.name.replace("_0.2_summary_3runs.csv", ""),
        "rmse_mean": df.loc["mean", "rmse_total"],
        "rmse_std": df.loc["std", "rmse_total"],
        "violation_mean": df.loc["mean", "violation"],
        "violation_std": df.loc["std", "violation"],
        "cb1_cb2_l2_mean": df.loc["mean", "cb1_cb2_l2"],
        "ca_weight": df.loc["mean", "ca_weight"],
        "cb1_weight": df.loc["mean", "cb1_weight"],
        "cb2_weight": df.loc["mean", "cb2_weight"],
        "cc_weight": df.loc["mean", "cc_weight"],
    }
    rows.append(row)

ranked = pd.DataFrame(rows)
ranked = ranked.sort_values("violation_mean")

out = base / "grid_ranked_by_violation.csv"
ranked.to_csv(out, index=False)

print(ranked.head(15))
print(f"\nSaved: {out}")