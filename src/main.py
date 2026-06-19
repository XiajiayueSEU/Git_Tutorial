import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Sample inference entry point for the course submission."
    )
    parser.add_argument(
        "--model_dir",
        default="outputs/imagenet_s_animals_10cls_trainval_holdout_ade/best_model",
        help="Trained model directory.",
    )
    parser.add_argument(
        "--input",
        default="data/test_examples",
        help="Path to one image or a directory of images.",
    )
    parser.add_argument(
        "--output",
        default="results/demo_inference",
        help="Directory to save inference outputs.",
    )
    args = parser.parse_args()

    import sys

    old_argv = sys.argv[:]
    sys.argv = [
        "infer",
        "--model_dir",
        args.model_dir,
        "--input",
        args.input,
        "--output",
        args.output,
    ]
    try:
        from .infer import main as infer_main
        infer_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
