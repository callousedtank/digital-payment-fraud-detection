import argparse

from src.experiment_tracking import load_experiments


def main():
    parser = argparse.ArgumentParser(description="Compare recorded model experiments.")
    parser.add_argument("experiment_file", help="Path to an experiment JSONL file.")
    arguments = parser.parse_args()

    experiments = sorted(
        load_experiments(arguments.experiment_file),
        key=lambda run: run["validation_metrics"]["accuracy"],
        reverse=True,
    )
    print("model_version\tmodel_type\taccuracy\trun_id")
    for run in experiments:
        print(
            f"{run['model_version']}\t{run['model_type']}\t"
            f"{run['validation_metrics']['accuracy']:.4f}\t{run['run_id']}"
        )


if __name__ == "__main__":
    main()
