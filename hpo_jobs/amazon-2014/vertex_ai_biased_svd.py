"""Hyperparameter optimization job for BiasedSVD on Amazon 2014 dataset using Vertex AI."""

import os

from google.cloud.aiplatform import hyperparameter_tuning as hpt

from hpo_jobs.vertex_ai_job import VertexAIHPOJob


def main():
    """Example usage of VertexAIHPOJob."""
    bucket = os.getenv(VertexAIHPOJob.Environment.BUCKET.value)
    if not bucket:
        raise EnvironmentError("BUCKET environment variable is not set.")

    bucket_uri = f"gs://{bucket}"
    amazon_data_path = f"{bucket_uri}/data/amazon-2014"
    amazon_split_data_path = f"{amazon_data_path}/split"

    hpo_job = VertexAIHPOJob(
        model_name="BiasedSVD",
        all_interactions_csv_path=f"{amazon_data_path}/all_interactions.csv",
        train_interactions_csv_path=f"{amazon_split_data_path}/train.csv",
        val_interactions_csv_path=f"{amazon_split_data_path}/val.csv",
        test_interactions_csv_path=f"{amazon_split_data_path}/test_neg_samples.csv",
    )

    param_spec = {
        "embedding_dim": hpt.DiscreteParameterSpec(
            values=[32, 64, 128, 256], scale="linear"
        ),
    }

    hpo_job.run(param_spec)


if __name__ == "__main__":
    main()
