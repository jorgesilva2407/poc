"""
Module responsible for calling the training and evaluation of the models implemented in the package.
"""

from argparse import ArgumentParser

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.losses.bpr import BPRLoss
from src.trainer import TrainerBuilder
from src.metrics.ndcg_at_k import NDCGAtK
from src.loggers.logger import LoggerBuilder
from src.models.recommender import Recommender
from src.models.biased_svd import BiasedSVDBuilder
from src.models.recommender import RecommenderBuilder
from src.loggers.tensorboard_logger import TensorBoardLoggerBuilder
from src.artifacts_saver.artifacts_saver import ArtifactsSaverBuilder
from src.datasets.precomputed_test_dataset import PrecomputedTestDataset
from src.datasets.negative_sampling_dataset import NegativeSamplingDataset
from src.datasets.recommendation_dataset import RecommendationDataset, TripletSample
from src.artifacts_saver.google_cloud_artifact_saver import (
    GoogleCloudArtifactSaverBuilder,
)


MODEL_BUILDER_REGISTRY: dict[str, RecommenderBuilder] = {
    "BiasedSVD": BiasedSVDBuilder(),
}

LOGGER_BUILDER_REGISTRY: dict[str, LoggerBuilder] = {
    "TensorBoard": TensorBoardLoggerBuilder(),
}

ARTIFACT_SAVER_REGISTRY: dict[str, ArtifactsSaverBuilder] = {
    "GoogleCloud": GoogleCloudArtifactSaverBuilder(),
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
    parser.add_argument(
        "--logger",
        type=str,
        default="TensorBoard",
        help="Name of the logger to use.",
        choices=LOGGER_BUILDER_REGISTRY.keys(),
    )
    parser.add_argument(
        "--artifacts-saver",
        type=str,
        default="GoogleCloud",
        help="Name of the artifacts saver to use.",
        choices=ARTIFACT_SAVER_REGISTRY.keys(),
    )
    known_args, remaining_args = parser.parse_known_args()
    model_name = known_args.model

    model_params, remaining_args = parse_model_args(model_name, remaining_args)
    optimizer_params, remaining_args = parse_optimizer_args(remaining_args)
    dataloader_params, remaining_args = parse_dataloader_args(remaining_args)

    print("Ignored arguments:", remaining_args)

    return model_name, model_params, optimizer_params, dataloader_params


def load_dataframes(
    dataloader_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataframes from CSV files specified in dataloader parameters.

    Args:
        dataloader_params (dict): Dictionary containing dataloader parameters.

    Returns:
        Tuple containing all interactions, training, validation, and test dataframes.
    """
    all_df = pd.read_csv(dataloader_params["all_interactions_csv"])
    train_df = pd.read_csv(dataloader_params["train_interactions_csv"])
    validation_df = pd.read_csv(
        dataloader_params["validation_interactions_csv"]
    ).sample(
        frac=dataloader_params["validation_subset_ratio"],
    )
    test_df = pd.read_csv(dataloader_params["test_interactions_csv"])
    return all_df, train_df, validation_df, test_df


def build_model(
    model_name: str,
    model_params: dict,
    num_users: int,
    num_items: int,
) -> Recommender:
    """
    Build the model based on the model name and parameters.

    Args:
        model_name (str): Name of the model to build.
        model_params (dict): Dictionary containing model parameters.
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.

    Returns:
        Recommender: Instantiated model.
    """
    model_builder = MODEL_BUILDER_REGISTRY[model_name]
    model = model_builder.with_num_users_items(
        num_users=num_users,
        num_items=num_items,
    ).build(model_params)
    return model


def build_datasets(
    all_df, train_df, validation_df, test_df
) -> tuple[RecommendationDataset, RecommendationDataset, PrecomputedTestDataset]:
    """
    Build datasets for training, validation, and testing.

    Args:
        all_df (pd.DataFrame): DataFrame containing all interactions.
        train_df (pd.DataFrame): DataFrame containing training interactions.
        validation_df (pd.DataFrame): DataFrame containing validation interactions.
        test_df (pd.DataFrame): DataFrame containing test interactions.

    Returns:
        Tuple containing training, validation, and test datasets.
    """
    train_dataset = NegativeSamplingDataset(
        interactions=train_df,
        all_interactions=all_df,
        num_negatives=1,
    )

    validation_dataset = NegativeSamplingDataset(
        interactions=validation_df,
        all_interactions=all_df,
        num_negatives=99,
    )

    test_dataset = PrecomputedTestDataset(
        test_df=test_df,
    )

    return train_dataset, validation_dataset, test_dataset


def build_data_loaders(
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> tuple[
    DataLoader[TripletSample], DataLoader[TripletSample], DataLoader[TripletSample]
]:
    """
    Build data loaders for training, validation, and testing.

    Args:
        all_df (pd.DataFrame): DataFrame containing all interactions.
        train_df (pd.DataFrame): DataFrame containing training interactions.
        validation_df (pd.DataFrame): DataFrame containing validation interactions.
        test_df (pd.DataFrame): DataFrame containing test interactions.
        batch_size (int): Batch size for the data loaders.

    Returns:
        Tuple containing training, validation, and test data loaders.
    """
    train_dataset, validation_dataset, test_dataset = build_datasets(
        all_df=all_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


def main():
    """
    Main function to parse arguments and print them.
    """
    model_name, model_params, optimizer_params, dataloader_params = parse_args()

    all_df, train_df, validation_df, test_df = load_dataframes(dataloader_params)
    batch_size = dataloader_params["batch_size"]
    num_users = all_df["user_id"].nunique()
    num_items = all_df["item_id"].nunique()

    model = build_model(
        model_name=model_name,
        model_params=model_params,
        num_users=num_users,
        num_items=num_items,
    )

    train_loader, validation_loader, test_loader = build_data_loaders(
        all_df=all_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        batch_size=batch_size,
    )

    metrics = [NDCGAtK(k=5), NDCGAtK(k=10)]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    trainer = (
        TrainerBuilder()
        .with_model(model)
        .with_data_loaders(
            train_loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
        )
        .with_optimizer(
            Adam(
                params=model.parameters(),
                lr=optimizer_params["learning_rate"],
                weight_decay=optimizer_params["weight_decay"],
            )
        )
        .with_loss(BPRLoss())
        .with_metrics(metrics)
        .with_early_stopping(
            metric=metrics[0],
            delta=0.001,
            maximize=True,
            patience=5,
        )
        .with_logger_builder(logger_builder=TensorBoardLoggerBuilder())
        .with_artifacts_saver_builder(
            artifacts_saver_builder=GoogleCloudArtifactSaverBuilder()
        )
        .with_device(device=device)
    ).build()

    trainer.run()

    # trainer = TrainerBuilder()
    #     .with_device()
    #     .build()


if __name__ == "__main__":
    main()
