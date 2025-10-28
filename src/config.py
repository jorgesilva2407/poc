"""Configuration module for the EnhancedGCR package."""

import os
from enum import StrEnum


class ENVIRONMENT:
    """Enumeration of environment variables used in the EnhancedGCR package."""

    class VARIABLES(StrEnum):
        """Enumeration of environment variable names."""

        TENSORBOARD_LOG_DIR = "TENSORBOARD_LOG_DIR"
        GCP_BUCKET_NAME = "GCP_BUCKET_NAME"
        GCP_BLOB_BASE_PATH = "GCP_BLOB_BASE_PATH"

    @staticmethod
    def get(var_name: "ENVIRONMENT") -> any:
        """Get the value of the environment variable."""
        return os.getenv(var_name.value)
