import argparse
import yaml
from engine.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="ExplainMyModel — ML Audit & Explainability Engine"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path or URL to input CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Model type to use"
    )

       parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml"
    )



    parser.add_argument(
        "--out",
        default="artifacts",
        help="Output directory for artifacts"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["models"].get(args.model, {})

    print("\n Running ExplainMyModel Pipeline")
    print(f" Dataset: {args.csv}")
    print(f" Target: {args.target}")
    print(f" Model: {args.model}")
    print(f" Output: {args.out}\n")

    payload = run_pipeline(
        csv_source=args.csv,
        target=args.target,
        model_type=args.model,
        model_config=model_config,
        output_dir=args.out
    )

    print("✅ Pipeline completed successfully.")
    print(f"📄 Results saved to `{args.out}/payload.json`")

if __name__ == "__main__":
    main()


