"""Evaluation script for aggregating and visualizing experiment results."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    """Parse command line arguments."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Script expects standard argparse format but workflow uses Hydra-style key=value
    # [CAUSE]: Workflow calls with results_dir="..." run_ids="..." but argparse expects positional args
    # [FIX]: Parse Hydra-style key=value arguments manually before using argparse
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    # parser.add_argument("results_dir", type=str, help="Results directory path")
    # parser.add_argument("run_ids", type=str, help="JSON string list of run IDs to compare")
    # ...
    # return parser.parse_args()
    #
    # [NEW CODE]:
    import sys

    # Check if arguments are in Hydra-style format (key=value)
    hydra_args = {}
    remaining_args = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            key, value = arg.split("=", 1)
            hydra_args[key] = value
        else:
            remaining_args.append(arg)

    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")

    if hydra_args:
        # Hydra-style arguments detected, use them directly
        parser.add_argument(
            "--wandb-entity", type=str, default="airas", help="WandB entity"
        )
        parser.add_argument(
            "--wandb-project",
            type=str,
            default="2026-0227-matsuzawa-2",
            help="WandB project",
        )
        args = parser.parse_args(remaining_args)
        # Add Hydra-style args to namespace
        args.results_dir = hydra_args.get("results_dir", ".research/results")
        args.run_ids = hydra_args.get("run_ids", "")
        return args
    else:
        # Standard argparse format
        parser.add_argument("results_dir", type=str, help="Results directory path")
        parser.add_argument(
            "run_ids", type=str, help="JSON string list of run IDs to compare"
        )
        parser.add_argument(
            "--wandb-entity", type=str, default="airas", help="WandB entity"
        )
        parser.add_argument(
            "--wandb-project",
            type=str,
            default="2026-0227-matsuzawa-2",
            help="WandB project",
        )
        return parser.parse_args()


def fetch_run_from_wandb(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dict with 'history', 'summary', and 'config'
    """
    print(f"Fetching run from WandB: {run_id}")

    api = wandb.Api()

    # Find runs with matching display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        # Try with -sanity suffix for sanity check runs
        runs = api.runs(
            f"{entity}/{project}-sanity",
            filters={"display_name": run_id},
            order="-created_at",
        )

    if not runs:
        raise ValueError(f"No run found with display_name: {run_id}")

    # Get most recent run
    run = runs[0]

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: SummarySubDict objects from wandb API are not JSON serializable
    # [CAUSE]: dict(run.summary) creates a dict with nested SummarySubDict values
    # [FIX]: Recursively convert to plain dict and handle special numeric types
    #
    # [OLD CODE]:
    # return {
    #     "id": run.id,
    #     "name": run.name,
    #     "history": run.history(),
    #     "summary": dict(run.summary),
    #     "config": dict(run.config),
    # }
    #
    # [NEW CODE]:
    def convert_to_json_serializable(obj):
        """Recursively convert wandb objects to JSON-serializable types."""
        import math

        # Handle special numeric types
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)  # Convert NaN/Inf to string
            return obj
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, bool, type(None))):
            return obj
        elif hasattr(obj, "__dict__"):
            # Convert objects with __dict__ to dict
            return convert_to_json_serializable(obj.__dict__)
        else:
            # Fallback: convert to string for unknown types
            return str(obj)

    return {
        "id": run.id,
        "name": run.name,
        "history": run.history(),
        "summary": convert_to_json_serializable(dict(run.summary)),
        "config": convert_to_json_serializable(dict(run.config)),
    }


def export_per_run_metrics(run_id: str, run_data: Dict[str, Any], results_dir: Path):
    """
    Export per-run metrics to JSON.

    Args:
        run_id: Run identifier
        run_data: Run data from WandB
        results_dir: Base results directory
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export summary metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(run_data["summary"], f, indent=2)

    print(f"Exported metrics: {metrics_file}")


def create_per_run_figures(run_id: str, run_data: Dict[str, Any], results_dir: Path):
    """
    Create per-run visualizations.

    Args:
        run_id: Run identifier
        run_data: Run data from WandB
        results_dir: Base results directory
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = run_data["summary"]

    # Create bar chart of key metrics
    metrics_to_plot = {
        "Accuracy (All)": summary.get("accuracy_all", 0),
        "Accuracy (Answered)": summary.get("accuracy_answered", 0),
        "Abstain Rate": summary.get("abstain_rate", 0),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    ax.set_ylabel("Value")
    ax.set_title(f"Key Metrics: {run_id}")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = run_dir / f"{run_id}_metrics.pdf"
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Created figure: {fig_path}")


def export_aggregated_metrics(
    run_ids: List[str], runs_data: Dict[str, Dict[str, Any]], results_dir: Path
) -> Dict[str, Any]:
    """
    Export aggregated comparison metrics.

    Args:
        run_ids: List of run identifiers
        runs_data: Dict mapping run_id to run data
        results_dir: Base results directory

    Returns:
        Aggregated metrics dict
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Determine primary metric
    primary_metric = "accuracy_answered"

    # Collect metrics by run_id
    metrics_by_run = {}
    for run_id in run_ids:
        metrics_by_run[run_id] = runs_data[run_id]["summary"]

    # Identify best proposed and best baseline
    proposed_runs = [r for r in run_ids if "proposed" in r]
    baseline_runs = [r for r in run_ids if "baseline" in r or "comparative" in r]

    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed = max(
            proposed_runs, key=lambda r: metrics_by_run[r].get(primary_metric, 0)
        )

    if baseline_runs:
        best_baseline = max(
            baseline_runs, key=lambda r: metrics_by_run[r].get(primary_metric, 0)
        )

    # Compute gap
    gap = None
    if best_proposed and best_baseline:
        gap = metrics_by_run[best_proposed].get(primary_metric, 0) - metrics_by_run[
            best_baseline
        ].get(primary_metric, 0)

    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    # Save to JSON
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Exported aggregated metrics: {agg_file}")

    return aggregated


def create_comparison_figures(
    run_ids: List[str], runs_data: Dict[str, Dict[str, Any]], results_dir: Path
):
    """
    Create comparison visualizations across runs.

    Args:
        run_ids: List of run identifiers
        runs_data: Dict mapping run_id to run data
        results_dir: Base results directory
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(run_ids))

    # Common metrics to compare
    common_metrics = [
        "accuracy_all",
        "accuracy_answered",
        "abstain_rate",
    ]

    # Create bar chart for each common metric
    for metric in common_metrics:
        values = []
        labels = []

        for run_id in run_ids:
            value = runs_data[run_id]["summary"].get(metric, 0)
            values.append(value)
            labels.append(run_id)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Comparison: {metric.replace('_', ' ').title()}")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fig_path = comparison_dir / f"comparison_{metric}.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close()

        print(f"Created comparison figure: {fig_path}")

    # Create summary comparison table figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    headers = [
        "Run ID",
        "Accuracy (All)",
        "Accuracy (Answered)",
        "Abstain Rate",
        "Total Samples",
    ]

    for run_id in run_ids:
        summary = runs_data[run_id]["summary"]
        row = [
            run_id,
            f"{summary.get('accuracy_all', 0):.3f}",
            f"{summary.get('accuracy_answered', 0):.3f}",
            f"{summary.get('abstain_rate', 0):.3f}",
            f"{int(summary.get('total_samples', 0))}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Results Summary Comparison", fontsize=14, weight="bold", pad=20)

    fig_path = comparison_dir / "comparison_summary_table.pdf"
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Created summary table: {fig_path}")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    results_dir = Path(args.results_dir)

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: json.loads fails when args.run_ids is an empty string or invalid JSON
    # [CAUSE]: The workflow may pass an empty string when run_ids input is not provided
    # [FIX]: Add error handling and auto-discover run directories when run_ids is empty/invalid
    #
    # [OLD CODE]:
    # run_ids = json.loads(args.run_ids)
    #
    # [NEW CODE]:
    try:
        if not args.run_ids or args.run_ids.strip() == "":
            # Auto-discover run directories in results_dir
            print("[INFO] No run_ids provided. Auto-discovering run directories...")
            run_dirs = [
                d.name
                for d in results_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            run_ids = sorted(run_dirs)
            if not run_ids:
                print(f"[ERROR] No run directories found in {results_dir}")
                print("[INFO] Available files:")
                for item in results_dir.iterdir():
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                raise ValueError(f"No run directories found in {results_dir}")
        else:
            run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse run_ids as JSON: {e}")
        print(f"[ERROR] Received: {repr(args.run_ids)}")
        raise ValueError(
            f'run_ids must be a valid JSON array (e.g., \'["run-1", "run-2"]\'), got: {repr(args.run_ids)}'
        )

    print("\n" + "=" * 80)
    print(f"Evaluation Script")
    print(f"Results dir: {results_dir}")
    print(f"Run IDs: {run_ids}")
    print("=" * 80)

    # Fetch data from WandB for each run
    runs_data = {}
    for run_id in run_ids:
        try:
            run_data = fetch_run_from_wandb(
                args.wandb_entity, args.wandb_project, run_id
            )
            runs_data[run_id] = run_data
        except Exception as e:
            print(f"[WARNING] Could not fetch run {run_id}: {e}")
            # Try to load from local files
            metrics_file = results_dir / run_id / "metrics.json"
            if metrics_file.exists():
                print(f"[INFO] Loading metrics from local file: {metrics_file}")
                with open(metrics_file) as f:
                    summary = json.load(f)
                runs_data[run_id] = {
                    "summary": summary,
                    "history": pd.DataFrame(),
                    "config": {},
                }
            else:
                print(f"[ERROR] No local metrics found for {run_id}")
                continue

    if not runs_data:
        print("\n" + "=" * 80)
        print("[ERROR] No run data available for visualization")
        print("=" * 80)
        print("\nPossible reasons:")
        print("1. No run_ids were provided and no run directories were found")
        print("2. All specified runs failed to load from WandB and local files")
        print(f"3. The results directory may be empty: {results_dir}")
        print("\nTo fix:")
        print("- Ensure you have completed main runs before visualization")
        print("- Check that run_ids parameter contains valid run identifiers")
        print(f"- Verify that run directories exist in {results_dir}")
        return 1

    # Export per-run metrics and figures
    for run_id, run_data in runs_data.items():
        export_per_run_metrics(run_id, run_data, results_dir)
        create_per_run_figures(run_id, run_data, results_dir)

    # Export aggregated metrics
    aggregated = export_aggregated_metrics(run_ids, runs_data, results_dir)

    # Create comparison figures
    create_comparison_figures(run_ids, runs_data, results_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in: {results_dir}")
    print(f"- Per-run metrics: {results_dir}/<run_id>/metrics.json")
    print(f"- Per-run figures: {results_dir}/<run_id>/*.pdf")
    print(f"- Aggregated metrics: {results_dir}/comparison/aggregated_metrics.json")
    print(f"- Comparison figures: {results_dir}/comparison/*.pdf")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Primary metric: {aggregated['primary_metric']}")
    if aggregated["best_proposed"]:
        print(
            f"Best proposed: {aggregated['best_proposed']} = {aggregated['metrics_by_run'][aggregated['best_proposed']][aggregated['primary_metric']]:.3f}"
        )
    if aggregated["best_baseline"]:
        print(
            f"Best baseline: {aggregated['best_baseline']} = {aggregated['metrics_by_run'][aggregated['best_baseline']][aggregated['primary_metric']]:.3f}"
        )
    if aggregated["gap"] is not None:
        print(f"Gap: {aggregated['gap']:.3f}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
