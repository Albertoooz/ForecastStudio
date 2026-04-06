"""
CLI entry point for the Forecaster pipeline.

Usage:
    python -m forecaster.cli --data sales.csv --target revenue
    python -m forecaster.cli --data sales.csv --target revenue --datetime date --horizon 30
    python -m forecaster.cli --data sales.csv --target revenue --auto  # skip confirmations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from forecaster.core.context import ContextWindow, ResourceBudget
from forecaster.core.pipeline import ForecastOrchestrator
from forecaster.utils.monitoring import ForecastMonitor, compute_cost


def _print_step(step, verbose: bool = False) -> None:
    """Pretty-print a pipeline step."""
    d = step.decision
    status = "✓" if step.success else "✗"
    if step.requires_confirmation:
        status = "⚠️  [AWAITING CONFIRMATION]"

    indent = "├─"
    print(f"  {indent} {d.agent_name}: {d.action} (confidence: {d.confidence:.2f}) {status}")

    if d.reasoning:
        print(f"  │   └─ {d.reasoning}")

    if verbose and d.parameters:
        for k, v in d.parameters.items():
            if k != "all_suggestions":
                print(f"  │      {k}: {v}")


def _print_audit_trail(context: ContextWindow) -> None:
    """Print the final audit trail."""
    print("\n" + "=" * 60)
    print(f"  AUDIT TRAIL — trace_id: {context.trace_id[:8]}")
    print("=" * 60)

    for i, d in enumerate(context.decision_log):
        marker = "└─" if i == len(context.decision_log) - 1 else "├─"
        conf_tag = " ⚠️ CONFIRM" if d.requires_confirmation else ""
        print(f"  {marker} [{d.agent_name}] {d.action} (conf={d.confidence:.2f}){conf_tag}")
        if d.reasoning:
            print(f"  │   {d.reasoning}")

    # Cost
    cost = compute_cost(context)
    print(
        f"\n  Cost: ${cost['total_cost_usd']:.4f} "
        f"(compute: ${cost['compute_cost_usd']:.4f}, LLM: ${cost['llm_cost_usd']:.4f})"
    )
    print(
        f"  Resources: {context.budget.consumed_memory_mb}MB / {context.budget.memory_budget_mb}MB, "
        f"{context.budget.consumed_compute_seconds:.1f}s / {context.budget.compute_budget_seconds}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forecaster",
        description="AI-powered time series forecasting pipeline",
    )
    parser.add_argument("--data", required=True, help="Path to CSV/Excel/Parquet data file")
    parser.add_argument("--target", required=True, help="Target column to forecast")
    parser.add_argument(
        "--datetime", default=None, help="Datetime column (auto-detected if omitted)"
    )
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (default: 7)")
    parser.add_argument(
        "--gap", type=int, default=0, help="Gap between last data point and first forecast"
    )
    parser.add_argument("--groups", nargs="*", default=[], help="Group-by columns for multi-series")
    parser.add_argument("--memory-mb", type=int, default=512, help="Memory budget in MB")
    parser.add_argument("--compute-s", type=float, default=300.0, help="Compute budget in seconds")
    parser.add_argument("--auto", action="store_true", help="Auto-approve all confirmations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", default=None, help="Output file for forecast (JSON)")
    parser.add_argument("--skip-external", action="store_true", help="Skip external data agent")

    args = parser.parse_args()

    # Validate input file
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        sys.exit(1)

    # Load data
    print(f"\n[FORECASTER] Loading data from {data_path}...")
    try:
        if data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
        elif data_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(data_path)
        elif data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Build context
    context = ContextWindow(
        budget=ResourceBudget(
            memory_budget_mb=args.memory_mb,
            compute_budget_seconds=args.compute_s,
        ),
        target_column=args.target,
        datetime_column=args.datetime,
        group_columns=args.groups,
        horizon=args.horizon,
        gap=args.gap,
        source_file=str(data_path),
    )
    context.register_data("primary", df)

    # Initialize monitor
    monitor = ForecastMonitor()

    # Build orchestrator
    orchestrator = ForecastOrchestrator(skip_external_data=args.skip_external)

    # Run pipeline
    print(f"\n[TRACE {context.trace_id[:8]}] Forecast request started")
    print(f"  Budget: {args.memory_mb}MB memory, {args.compute_s}s compute")
    print(f"  Target: {args.target}, Horizon: {args.horizon}\n")

    gen = orchestrator.run(context)
    try:
        step = next(gen)
        while True:
            _print_step(step, verbose=args.verbose)

            # Log to monitor
            monitor.on_decision(step.context, step.decision)

            if step.requires_confirmation:
                if args.auto:
                    print("  │   └─ Auto-approved")
                    step = gen.send("approve")
                else:
                    # Interactive mode
                    print("\n  Options: [a]pprove / [s]kip / [c]ancel")
                    user_input = input("  > ").strip().lower()
                    if user_input in ("a", "approve", "yes", "tak"):
                        step = gen.send("approve")
                    elif user_input in ("c", "cancel", "no", "nie"):
                        print("  Pipeline cancelled by user.")
                        break
                    else:
                        step = gen.send("skip")
            else:
                step = next(gen)

    except StopIteration as e:
        context = e.value if e.value else step.context

    # Summary
    monitor.on_pipeline_complete(context)
    _print_audit_trail(context)

    # Output forecast
    if context.forecast_result:
        print("\n  Forecast completed successfully!")
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(context.forecast_result, f, indent=2, default=str)
            print(f"  Results saved to: {output_path}")
        elif args.verbose:
            print(
                f"  Result: {json.dumps(context.forecast_result, indent=2, default=str)[:500]}..."
            )
    else:
        print("\n  No forecast produced. Check audit trail above.")

    monitor.close()


if __name__ == "__main__":
    main()
