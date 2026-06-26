from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .train import train_model
from .predictor import predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def process_sources(config_section: dict[str, Any]) -> pd.DataFrame:
    output_df = pd.DataFrame()
    sources = config_section.get("sources", [])
    
    for source in sources:
        file_path = PROJECT_ROOT / source["file"]
        if not file_path.exists():
            print(f"Warning: Source file not found: {file_path}")
            continue
            
        print(f"Processing {file_path}...")
        
        # Load data based on extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format for {file_path}")
            continue

        # Apply filters
        filters = source.get("filters", {})
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]

        # Rename columns
        mapping = source.get("mapping", {})
        if mapping:
            df = df.rename(columns=mapping)

        # Apply default values for missing columns
        defaults = source.get("defaults", {})
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
                
        # Keep only the mapped columns and default columns if they exist
        expected_cols = list(mapping.values()) + list(defaults.keys())
        # Also include any columns that might already be named correctly
        # But to be safe and avoid clutter, we filter columns to those we care about.
        # Let's get the final target columns if defined, else just concatenate
        
        output_df = pd.concat([output_df, df], ignore_index=True)
        
    return output_df


def run_pipeline(config_path: Path) -> None:
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Process matches
    if "matches" in config:
        print("Processing match data...")
        matches_df = process_sources(config["matches"])
        if not matches_df.empty:
            out_path = PROJECT_ROOT / config["matches"]["output"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Retain only required columns if possible, but for flexibility we save all
            matches_df.to_csv(out_path, index=False)
            print(f"Saved match data to {out_path} ({len(matches_df)} rows)")

    # Process players
    if "players" in config:
        print("\nProcessing player data...")
        players_df = process_sources(config["players"])
        if not players_df.empty:
            out_path = PROJECT_ROOT / config["players"]["output"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            players_df.to_csv(out_path, index=False)
            print(f"Saved player data to {out_path} ({len(players_df)} rows)")

    # Train model
    print("\nTraining model with new data...")
    try:
        # Default paths match what's in train.py but we can override if needed
        matches_path = PROJECT_ROOT / config.get("matches", {}).get("output", "data/sample_matches.csv")
        players_path = PROJECT_ROOT / config.get("players", {}).get("output", "data/sample_players.csv")
        
        artifact = train_model(matches_path=matches_path, players_path=players_path)
        print("Model training completed successfully.")
        print("Validation metrics:")
        for key, value in artifact["metrics"].items():
            print(f"  {key}: {value:.3f}")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)


def run_crawling_pipeline(
    matches_url: str | None,
    players_url: str | None,
    matches_out: Path,
    players_out: Path,
) -> None:
    from .crawler import crawl_and_process
    
    print("Starting web crawling pipeline...")
    matches_df, players_df = crawl_and_process(matches_url, players_url)
    
    # Save results to files
    if matches_df is not None and not matches_df.empty:
        matches_out.parent.mkdir(parents=True, exist_ok=True)
        matches_df.to_csv(matches_out, index=False)
        print(f"Saved crawled matches to {matches_out} ({len(matches_df)} rows)")
    else:
        matches_df = matches_out if matches_out.exists() else None
        
    if players_df is not None and not players_df.empty:
        players_out.parent.mkdir(parents=True, exist_ok=True)
        players_df.to_csv(players_out, index=False)
        print(f"Saved crawled players to {players_out} ({len(players_df)} rows)")
    else:
        players_df = players_out if players_out.exists() else None
        
    print("\nTraining model with crawled data...")
    try:
        artifact = train_model(matches_path=matches_df, players_path=players_df)
        print("Model training completed successfully.")
        print("Validation metrics:")
        for key, value in artifact["metrics"].items():
            print(f"  {key}: {value:.3f}")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Football AI Data Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="pipeline_config.json", 
        help="Path to pipeline configuration JSON file"
    )
    parser.add_argument(
        "--matches-url",
        type=str,
        help="URL to scrape match results from"
    )
    parser.add_argument(
        "--players-url",
        type=str,
        help="URL to scrape players squad statistics from"
    )
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME_TEAM", "AWAY_TEAM"),
        help="Run a prediction after pipeline finishes"
    )
    
    args = parser.parse_args()
    
    if args.matches_url or args.players_url:
        config_path = PROJECT_ROOT / args.config
        matches_out = PROJECT_ROOT / "data" / "sample_matches.csv"
        players_out = PROJECT_ROOT / "data" / "sample_players.csv"
        
        # Try to read outputs from config if present
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "matches" in config and "output" in config["matches"]:
                        matches_out = PROJECT_ROOT / config["matches"]["output"]
                    if "players" in config and "output" in config["players"]:
                        players_out = PROJECT_ROOT / config["players"]["output"]
            except Exception:
                pass
                
        run_crawling_pipeline(
            matches_url=args.matches_url,
            players_url=args.players_url,
            matches_out=matches_out,
            players_out=players_out
        )
    else:
        config_path = PROJECT_ROOT / args.config
        run_pipeline(config_path)

    if args.predict:
        print(f"\nRunning prediction for {args.predict[0]} vs {args.predict[1]}...")
        try:
            result = predict(args.predict[0], args.predict[1])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

