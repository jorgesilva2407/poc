"""
Module responsible for calling the training and evaluation of the models implemented in the package.
"""

from argparse import ArgumentParser

from src.models.recommender import RecommenderBuilder
from src.models.biased_svd import BiasedSVDBuilder


MODEL_BUILDER_REGISTRY: dict[str, RecommenderBuilder] = {
    "BiasedSVD": BiasedSVDBuilder(),
}


def parse_optimizer_args(remaining_args: list[str]) -> tuple[dict, list[str]]:
    """
    Parse optimizer arguments from the command line.

    Args:
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing a dictionary of optimizer arguments and the remaining arguments.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=True,
        help="Weight decay (L2 regularization) for the optimizer.",
    )
    optimizer_args, remaining_args = parser.parse_known_args(remaining_args)
    return vars(optimizer_args), remaining_args


def parse_dataloader_args(remaining_args: list[str]) -> tuple[dict, list[str]]:
    """
    Parse dataloader arguments from the command line.

    Args:
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing a dictionary of dataloader arguments and the remaining arguments.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--all-interactions-csv",
        type=str,
        required=True,
        help="Path to the CSV file containing all interactions.",
    )
    parser.add_argument(
        "--train-interactions-csv",
        type=str,
        required=True,
        help="Path to the CSV file containing positive interactions.",
    )
    parser.add_argument(
        "--validation-interactions-csv",
        type=str,
        required=True,
        help="Path to the CSV file containing validation interactions.",
    )
    parser.add_argument(
        "--validation-subset-ratio",
        type=float,
        default=0.1,
        help="Ratio of the validation set to use for validation.",
    )
    parser.add_argument(
        "--test-interactions-csv",
        type=str,
        required=True,
        help="Path to the CSV file containing test interactions.",
    )
    dataloader_args, remaining_args = parser.parse_known_args(remaining_args)
    return vars(dataloader_args), remaining_args


def parse_model_args(
    model_name: str, remaining_args: list[str]
) -> tuple[dict, list[str]]:
    """
    Parse model arguments from the command line.

    Args:
        model_name (str): Name of the model to parse arguments for.
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing a dictionary of model arguments and the remaining arguments.
    """
    parser = MODEL_BUILDER_REGISTRY[model_name].argparser
    model_args, remaining_args = parser.parse_known_args(remaining_args)
    return vars(model_args), remaining_args


def parse_args() -> tuple[str, dict, dict, dict]:
    """
    Parse command line arguments.

    Returns:
        Tuple containing model name, model parameters, optimizer parameters and dataloader
        parameters.
    """
    parser = ArgumentParser(description="EnhancedGCR Training and Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use.",
        choices=MODEL_BUILDER_REGISTRY.keys(),
    )
    known_args, remaining_args = parser.parse_known_args()
    model_name = known_args.model

    model_params, remaining_args = parse_model_args(model_name, remaining_args)
    optimizer_params, remaining_args = parse_optimizer_args(remaining_args)
    dataloader_params, remaining_args = parse_dataloader_args(remaining_args)

    print("Ignored arguments:", remaining_args)

    return model_name, model_params, optimizer_params, dataloader_params


def main():
    """
    Main function to parse arguments and print them.
    """
    model_name, model_params, optimizer_params, dataloader_params = parse_args()

    print("Model Name:", model_name)
    print("Model Parameters:", model_params)
    print("Optimizer Parameters:", optimizer_params)
    print("Dataloader Parameters:", dataloader_params)


if __name__ == "__main__":
    main()
