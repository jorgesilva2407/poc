"""
Module responsible for calling the training and evaluation of the models implemented in the package.
"""

import sys
import random
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.trainer import Trainer
from src.losses.bpr import BPRLoss
from src.metrics.ndcg_at_k import NDCGAtK
from src.loggers.logger import LoggerBuilder
from src.models.recommender import Recommender
from src.models.biased_svd import BiasedSVDFactory
from src.models.recommender import RecommenderFactory, Context
from src.loggers.tensorboard_logger import TensorBoardLoggerBuilder
from src.artifacts_savers.artifacts_saver import ArtifactsSaverBuilder
from src.datasets.precomputed_test_dataset import PrecomputedTestDataset
from src.datasets.negative_sampling_dataset import NegativeSamplingDataset
from src.experiment_trackers.experiment_tracker import ExperimentTrackerBuilder
from src.artifacts_savers.local_artifacts_saver import LocalArtifactsSaverBuilder
from src.datasets.recommendation_dataset import RecommendationDataset, TripletSample
from src.experiment_trackers.noop_experiment_tracker import NoOpExperimentTrackerBuilder
from src.artifacts_savers.google_cloud_artifact_saver import (
    GoogleCloudArtifactSaverBuilder,
)
from src.experiment_trackers.optuna_experiment_tracker import (
    OptunaExperimentTrackerBuilder,
)
from src.experiment_trackers.vertex_ai_experiment_tracker import (
    VertexAIExperimentTrackerBuilder,
)

SEED = 42

MODEL_FACTORY_REGISTRY: dict[str, RecommenderFactory] = {
    "BiasedSVD": BiasedSVDFactory(),
}

LOGGER_BUILDER_REGISTRY: dict[str, LoggerBuilder] = {
    "TensorBoard": TensorBoardLoggerBuilder(),
}

ARTIFACT_SAVER_REGISTRY: dict[str, ArtifactsSaverBuilder] = {
    "GoogleCloud": GoogleCloudArtifactSaverBuilder(),
    "Local": LocalArtifactsSaverBuilder(),
}

EXPERIMENT_TRACKER_BUILDER_REGISTRY: dict[str, ExperimentTrackerBuilder] = {
    "NoOp": NoOpExperimentTrackerBuilder(),
    "VertexAI": VertexAIExperimentTrackerBuilder(),
    "Optuna": OptunaExperimentTrackerBuilder(),
}


def set_seed():
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed (int): Random seed value.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def parse_model_args(
    model_name: str, remaining_args: list[str]
) -> tuple[RecommenderFactory, dict, list[str]]:
    """
    Parse model arguments from the command line.

    Args:
        model_name (str): Name of the model to parse arguments for.
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing a dictionary of model arguments and the remaining arguments.
    """
    parser = MODEL_FACTORY_REGISTRY[model_name].argparser
    model_args, remaining_args = parser.parse_known_args(remaining_args)
    model_factory = MODEL_FACTORY_REGISTRY[model_name]
    return model_factory, vars(model_args), remaining_args


def parse_logger_args(
    logger_name: str, args: list[str]
) -> tuple[LoggerBuilder, list[str]]:
    """
    Parse logger arguments from the command line.

    Args:
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing a LoggerBuilder instance and the remaining arguments.
    """
    parser = LOGGER_BUILDER_REGISTRY[logger_name].argparser
    logger_args, remaining_args = parser.parse_known_args(args)
    logger_builder = LOGGER_BUILDER_REGISTRY[logger_name].with_configuration(
        vars(logger_args)
    )
    return logger_builder, remaining_args


def parse_artifacts_saver_args(
    artifacts_saver_name: str, args: list[str]
) -> tuple[ArtifactsSaverBuilder, list[str]]:
    """
    Parse artifacts saver arguments from the command line.

    Args:
        remaining_args (list[str]): List of remaining command line arguments.

    Returns:
        Tuple containing an ArtifactsSaverBuilder instance and the remaining arguments.
    """
    parser = ARTIFACT_SAVER_REGISTRY[artifacts_saver_name].argparser
    artifacts_saver_args, remaining_args = parser.parse_known_args(args)
    artifacts_saver_builder = ARTIFACT_SAVER_REGISTRY[
        artifacts_saver_name
    ].with_configuration(vars(artifacts_saver_args))
    return artifacts_saver_builder, remaining_args


def parse_experiment_tracker_args(
    experiment_tracker_name: str, args: list[str]
) -> tuple[ExperimentTrackerBuilder, list[str]]:
    """
    Parse experiment tracker arguments from the command line.

    Args:
        remaining_args (list[str]): List of remaining command line arguments.
    Returns:
        Tuple containing an ExperimentTrackerBuilder instance and the remaining arguments.
    """
    parser = EXPERIMENT_TRACKER_BUILDER_REGISTRY[experiment_tracker_name].argparser
    experiment_tracker_args, remaining_args = parser.parse_known_args(args)
    experiment_tracker_builder = EXPERIMENT_TRACKER_BUILDER_REGISTRY[
        experiment_tracker_name
    ].with_configuration(vars(experiment_tracker_args))
    return experiment_tracker_builder, remaining_args


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
        help="Path to the CSV file containing test interactions and NEGATIVE SAMPLING.",
    )
    dataloader_args, remaining_args = parser.parse_known_args(remaining_args)
    return vars(dataloader_args), remaining_args


def parse_args() -> tuple[
    RecommenderFactory,
    dict,
    LoggerBuilder,
    ArtifactsSaverBuilder,
    ExperimentTrackerBuilder,
    dict,
    dict,
]:
    """
    Parse command line arguments.

    Returns:
        Tuple containing model factory, model hyper-parameters, logger builder,
        artifacts saver builder, optimizer parameters, and dataloader parameters.
    """
    parser = ArgumentParser(description="EnhancedGCR Training and Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use.",
        choices=MODEL_FACTORY_REGISTRY.keys(),
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
    parser.add_argument(
        "--experiment-tracker",
        type=str,
        default="NoOp",
        help="Name of the experiment tracker to use.",
        choices=EXPERIMENT_TRACKER_BUILDER_REGISTRY.keys(),
    )
    known_args, remaining_args = parser.parse_known_args()
    model_name = known_args.model
    logger_name = known_args.logger
    artifacts_saver_name = known_args.artifacts_saver
    experiment_tracker_name = known_args.experiment_tracker

    model_factory, model_params, remaining_args = parse_model_args(
        model_name, remaining_args
    )
    logger_builder, remaining_args = parse_logger_args(logger_name, remaining_args)
    artifacts_saver_builder, remaining_args = parse_artifacts_saver_args(
        artifacts_saver_name, remaining_args
    )
    experiment_tracker_builder, remaining_args = parse_experiment_tracker_args(
        experiment_tracker_name, remaining_args
    )
    optimizer_params, remaining_args = parse_optimizer_args(remaining_args)
    dataloader_params, remaining_args = parse_dataloader_args(remaining_args)

    print("Ignored arguments:", remaining_args, file=sys.stderr)

    return (
        model_factory,
        model_params,
        logger_builder,
        artifacts_saver_builder,
        experiment_tracker_builder,
        optimizer_params,
        dataloader_params,
    )


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
        random_state=SEED,
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
    model_factory = MODEL_FACTORY_REGISTRY[model_name]
    model = model_factory.create(num_users, num_items, model_params)
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

    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, validation_loader, test_loader


def main():
    """
    Main function to parse arguments and print them.
    """
    set_seed()

    (
        model_factory,
        model_params,
        logger_builder,
        artifacts_saver_builder,
        experiment_tracker_builder,
        optimizer_params,
        dataloader_params,
    ) = parse_args()

    all_df, train_df, validation_df, test_df = load_dataframes(dataloader_params)
    batch_size = dataloader_params["batch_size"]
    num_users = all_df["user_id"].nunique()
    num_items = all_df["item_id"].nunique()

    model_context = Context(
        num_users=num_users,
        num_items=num_items,
        interactions_df=train_df,
    )

    model = model_factory.create(
        context=model_context,
        args=model_params,
    )

    experiment_tracker = experiment_tracker_builder.build()
    all_hparams = {**model_params, **optimizer_params, **dataloader_params}
    experiment_tracker.log_params(all_hparams)

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

    print("Using device:", device, file=sys.stderr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        optimizer=Adam(
            params=model.parameters(),
            lr=optimizer_params["learning_rate"],
            weight_decay=optimizer_params["weight_decay"],
        ),
        epochs=50,
        loss=BPRLoss(),
        metrics=metrics,
        early_stopping_metric=metrics[0].name,
        early_stopping_delta=0.001,
        early_stopping_patience=5,
        maximize_metric=True,
        logger_builder=logger_builder,
        artifacts_saver_builder=artifacts_saver_builder,
        device=device,
    )

    run_results = trainer.run()

    experiment_tracker.log_metrics(run_results["test_metrics"])
    experiment_tracker.report_hpo_metric(
        metric_name=run_results["best_metric_name"],
        metric_value=run_results["best_val_metric"],
        global_step=run_results["best_epoch"],
    )


if __name__ == "__main__":
    main()
