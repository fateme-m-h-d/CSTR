# collect_linearization_accuracy.py

import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================
# CONFIG
# ============================

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11, 30, 90]
SEGMENTATION_TYPES = ["uniform", "adaptive"]

ADAPTIVE_SPLIT_METRIC = "mean_abs_f_pl"

OUT_ROOT = Path("linearization_accuracy_results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================
# HELPERS
# ============================

def run_cmd(cmd, log_path):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout)

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed. Check log: {log_path}")

    return result.stdout


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def make_linearization(segmentation_type, n_regions, scenario_dir):
    if segmentation_type == "uniform":
        run_cmd(
            [
                sys.executable,
                "linearization.py",
                "--nT_regions",
                str(n_regions),
            ],
            scenario_dir / "logs" / "linearization.log",
        )

    elif segmentation_type == "adaptive":
        run_cmd(
            [
                sys.executable,
                "adaptive_linearization.py",
                "--nT_regions",
                str(n_regions),
                "--split_metric",
                ADAPTIVE_SPLIT_METRIC,
            ],
            scenario_dir / "logs" / "adaptive_linearization.log",
        )

    else:
        raise ValueError(f"Unknown segmentation_type: {segmentation_type}")


def run_linearization_accuracy(scenario_dir):
    run_cmd(
        [
            sys.executable,
            "linearization_accuracy.py",
        ],
        scenario_dir / "logs" / "linearization_accuracy.log",
    )


def summarize_current_linearization(segmentation_type, n_regions):
    detailed = pd.read_csv("linearization_accuracy_detailed.csv")
    region_summary = pd.read_csv("linearization_accuracy_summary.csv")

    row = {
        "segmentation_type": segmentation_type,
        "n_regions": n_regions,

        # residual-based linearization accuracy
        "overall_mean_abs_linearization_error": detailed["abs_linearization_error"].mean(),
        "overall_rmse_linearization_error": np.sqrt(
            np.mean(detailed["linearization_error"] ** 2)
        ),
        "overall_max_abs_linearization_error": detailed["abs_linearization_error"].max(),

        # PL-y accuracy, scaled
        "overall_mean_abs_PL_y_error_scaled": detailed[
            [
                "abs_Ca_PL_error_scaled",
                "abs_Cb_PL_error_scaled",
                "abs_Cc_PL_error_scaled",
            ]
        ].to_numpy().mean(),

        "overall_rmse_PL_y_error_scaled": np.sqrt(
            np.mean(
                detailed[
                    [
                        "abs_Ca_PL_error_scaled",
                        "abs_Cb_PL_error_scaled",
                        "abs_Cc_PL_error_scaled",
                    ]
                ].to_numpy() ** 2
            )
        ),

        "overall_max_abs_PL_y_error_scaled": detailed[
            [
                "abs_Ca_PL_error_scaled",
                "abs_Cb_PL_error_scaled",
                "abs_Cc_PL_error_scaled",
            ]
        ].to_numpy().max(),

        # PL-y accuracy, original units
        "overall_mean_abs_PL_y_error_original": detailed[
            [
                "abs_Ca_PL_error",
                "abs_Cb_PL_error",
                "abs_Cc_PL_error",
            ]
        ].to_numpy().mean(),

        "overall_rmse_PL_y_error_original": np.sqrt(
            np.mean(
                detailed[
                    [
                        "abs_Ca_PL_error",
                        "abs_Cb_PL_error",
                        "abs_Cc_PL_error",
                    ]
                ].to_numpy() ** 2
            )
        ),

        "overall_max_abs_PL_y_error_original": detailed[
            [
                "abs_Ca_PL_error",
                "abs_Cb_PL_error",
                "abs_Cc_PL_error",
            ]
        ].to_numpy().max(),

        # nonlinear residual after PL projection
        "mean_abs_f_true": detailed["abs_f_true"].mean(),
        "mean_abs_f_PL": detailed["abs_f_PL"].mean(),
        "max_abs_f_PL": detailed["abs_f_PL"].max(),

        # concentration changes caused by PL projection
        "mean_abs_dCa": detailed["dCa_PL"].abs().mean(),
        "mean_abs_dCb": detailed["dCb_PL"].abs().mean(),
        "max_abs_dCa": detailed["dCa_PL"].abs().max(),
        "max_abs_dCb": detailed["dCb_PL"].abs().max(),

        # sensitivity / amplification
        "mean_abs_df_dCa": detailed["abs_df_dCa"].mean(),
        "mean_abs_df_dCb": detailed["abs_df_dCb"].mean(),
        "max_abs_df_dCa": detailed["abs_df_dCa"].max(),
        "max_abs_df_dCb": detailed["abs_df_dCb"].max(),

        "mean_amplification": detailed["amplification"].mean(),
        "max_amplification": detailed["amplification"].max(),

        # data/region diagnostics
        "total_points": len(detailed),
        "nonempty_regions": int((region_summary["n_points"] > 0).sum()),
        "empty_regions": int((region_summary["n_points"] == 0).sum()),
        "min_points_per_region": int(region_summary["n_points"].min()),
        "max_points_per_region": int(region_summary["n_points"].max()),
        "mean_points_per_region": float(region_summary["n_points"].mean()),
    }

    region_summary.insert(0, "segmentation_type", segmentation_type)
    region_summary.insert(1, "n_regions", n_regions)

    return row, region_summary


def archive_current_outputs(scenario_dir):
    files_to_copy = [
        "region_edges.npz",
        "lin_params.csv",
        "ABb_matrices.csv",
        "adaptive_segmentation_summary.csv",
        "linearization_accuracy_detailed.csv",
        "linearization_accuracy_summary.csv",
    ]

    for fname in files_to_copy:
        copy_if_exists(fname, scenario_dir / "artifacts" / fname)


def main():
    all_overall_rows = []
    all_region_rows = []

    for segmentation_type in SEGMENTATION_TYPES:
        for n_regions in SEGMENT_SCENARIOS:
            scenario_name = f"{segmentation_type}_S{n_regions}"
            scenario_dir = OUT_ROOT / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            print("\n\n" + "#" * 100)
            print(f"SCENARIO: {scenario_name}")
            print("#" * 100)

            # 1) Generate lin_params.csv, ABb_matrices.csv, region_edges.npz
            make_linearization(segmentation_type, n_regions, scenario_dir)

            # 2) Run linearization accuracy for this scenario
            run_linearization_accuracy(scenario_dir)

            # 3) Read current outputs and summarize
            overall_row, region_summary = summarize_current_linearization(
                segmentation_type,
                n_regions,
            )

            all_overall_rows.append(overall_row)
            all_region_rows.append(region_summary)

            # 4) Save current scenario files before they get overwritten
            archive_current_outputs(scenario_dir)

            # 5) Save partial combined files after every scenario
            overall_df = pd.DataFrame(all_overall_rows)
            overall_df.to_csv(
                OUT_ROOT / "linearization_accuracy_all_scenarios_partial.csv",
                index=False,
            )

            region_df = pd.concat(all_region_rows, ignore_index=True)
            region_df.to_csv(
                OUT_ROOT / "linearization_accuracy_all_regions_partial.csv",
                index=False,
            )

    # Final combined files
    overall_df = pd.DataFrame(all_overall_rows)
    overall_df.to_csv(
        OUT_ROOT / "linearization_accuracy_all_scenarios.csv",
        index=False,
    )

    region_df = pd.concat(all_region_rows, ignore_index=True)
    region_df.to_csv(
        OUT_ROOT / "linearization_accuracy_all_regions.csv",
        index=False,
    )

    print("\n\nDONE.")
    print(f"Saved: {OUT_ROOT / 'linearization_accuracy_all_scenarios.csv'}")
    print(f"Saved: {OUT_ROOT / 'linearization_accuracy_all_regions.csv'}")


if __name__ == "__main__":
    main()