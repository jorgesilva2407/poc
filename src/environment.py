"""Configuration module for the EnhancedGCR package."""

import os
from enum import StrEnum


class ENVIRONMENT:
    """Environment variable management."""

    class VARIABLES(StrEnum):
        """Enumeration of environment variable names."""

        TENSORBOARD_LOG_DIR = "TENSORBOARD_LOG_DIR"

    @staticmethod
    def get(var_name: "ENVIRONMENT") -> any:
        """Get the value of the environment variable."""
        return os.getenv(var_name.value)
