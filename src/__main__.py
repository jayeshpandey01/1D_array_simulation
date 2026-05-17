"""
Unified CLI for the 1D signal simulation project.

  python -m src generate
  python -m src analyze --csv datasets/test_parameters.csv --row 0
  python -m src evaluate --csv datasets/test_parameters.csv
  python -m src compare --csv datasets/test_parameters.csv --target frequency
  python -m src run-all
  python -m src gui
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="1D array signal simulation — unified entry point",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Create datasets/ train and test CSVs")

    p_analyze = sub.add_parser("analyze", help="Analyze one row from a CSV")
    p_analyze.add_argument("--csv", required=True)
    p_analyze.add_argument("--row", type=int, default=0)

    p_eval = sub.add_parser("evaluate", help="Batch evaluation + plots")
    p_eval.add_argument("--csv", default="datasets/test_parameters.csv")
    p_eval.add_argument("--output-dir", default="outputs/evaluation")
    p_eval.add_argument("--max-rows", type=int, default=None, help="Limit rows analyzed")
    p_eval.add_argument("--show", action="store_true", help="Show plots interactively")

    p_cmp = sub.add_parser("compare", help="ML vs FFT model comparison")
    p_cmp.add_argument("--csv", default="datasets/test_parameters.csv")
    p_cmp.add_argument("--target", default="frequency", choices=["frequency", "amplitude", "phase"])
    p_cmp.add_argument("--output-dir", default="outputs/comparison")
    p_cmp.add_argument("--cnn", action="store_true", help="Train PyTorch CNN (requires torch)")

    p_all = sub.add_parser("run-all", help="generate → evaluate → compare")
    p_all.add_argument("--csv", default="datasets/test_parameters.csv")
    p_all.add_argument("--skip-ml", action="store_true")

    sub.add_parser("gui", help="Open Tkinter analyzer GUI")

    p_db = sub.add_parser("db", help="SQLite database commands")
    db_sub = p_db.add_subparsers(dest="db_cmd", required=True)

    db_init = db_sub.add_parser("init", help="Create database tables")
    db_init.add_argument("--db", default="data/signals.db")

    db_import = db_sub.add_parser("import", help="Import CSV into database")
    db_import.add_argument("--csv", required=True)
    db_import.add_argument("--db", default="data/signals.db")

    db_list = db_sub.add_parser("list", help="List signals in database")
    db_list.add_argument("--db", default="data/signals.db")
    db_list.add_argument("--limit", type=int, default=10)

    db_analyze = db_sub.add_parser("analyze", help="Analyze one signal by DB id")
    db_analyze.add_argument("--id", type=int, required=True)
    db_analyze.add_argument("--db", default="data/signals.db")

    db_export = db_sub.add_parser("export", help="Export database to CSV")
    db_export.add_argument("--db", default="data/signals.db")
    db_export.add_argument("--out", default="datasets/from_database.csv")

    args = parser.parse_args(argv)

    if args.command == "generate":
        from src.pipeline import generate_datasets

        train_p, test_p = generate_datasets()
        print(f"Train: {train_p}")
        print(f"Test:  {test_p}")

    elif args.command == "analyze":
        from src.pipeline import analyze_csv
        from src.signal_io import load_single_signal
        from src.signal_report import format_report

        try:
            sig, gt = load_single_signal(args.csv, row_index=args.row)
        except FileNotFoundError:
            print(f"File not found: {args.csv}")
            print("Run: python -m src generate")
            sys.exit(1)
        from src.pipeline import analyze_one

        result = analyze_one(sig, ground_truth=gt)
        print(format_report(result))

    elif args.command == "evaluate":
        from src.pipeline import evaluate_csv

        evaluate_csv(
            args.csv,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            show_plots=args.show,
        )

    elif args.command == "compare":
        from src.pipeline import compare_models

        compare_models(
            args.csv,
            target_col=args.target,
            output_dir=args.output_dir,
            train_cnn=args.cnn,
        )

    elif args.command == "run-all":
        from src.pipeline import run_full

        run_full(csv_path=args.csv, skip_ml=args.skip_ml)

    elif args.command == "gui":
        from src.gui.app import main as gui_main

        gui_main()

    elif args.command == "db":
        from src.database import (
            DEFAULT_DB_PATH,
            analyze_and_store,
            export_to_csv,
            import_csv,
            init_db,
            list_signals,
        )
        from src.signal_report import format_report

        db = getattr(args, "db", DEFAULT_DB_PATH)

        if args.db_cmd == "init":
            path = init_db(db)
            print(f"Database ready: {path}")

        elif args.db_cmd == "import":
            n = import_csv(args.csv, db_path=db)
            print(f"Imported {n} signals into {db}")

        elif args.db_cmd == "list":
            rows = list_signals(db, limit=args.limit)
            if not rows:
                print("No signals. Run: python -m src db import --csv datasets/test_parameters.csv")
            for r in rows:
                print(
                    f"id={r['id']}  {r.get('wave_type') or '?'}  "
                    f"f={r.get('frequency')}  source={r.get('source_file')}"
                )

        elif args.db_cmd == "analyze":
            result = analyze_and_store(args.id, db_path=db)
            print(format_report(result))

        elif args.db_cmd == "export":
            out = export_to_csv(db_path=db, output_path=args.out)
            print(f"Exported to {out}")


if __name__ == "__main__":
    main()
