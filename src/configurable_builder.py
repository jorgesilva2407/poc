"""A base class for configurable builders."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import TypeVar, Generic, Dict

T = TypeVar("T")


class ConfigurableBuilder(ABC, Generic[T]):
    """
    A base class for building configurable components.
    """

    def __init__(self):
        self._cli_args = None

    @property
    @abstractmethod
    def argparser(self) -> ArgumentParser:
        """
        Returns the ArgumentParser instance for CLI argument definition.

        Returns:
            ArgumentParser: The argument parser for the builder.
        """
        return ArgumentParser(add_help=False)

    def with_configuration(self, args: Dict) -> "ConfigurableBuilder":
        """
        Configures the builder with the provided arguments.

        Args:
            args (Dict): A dictionary of configuration arguments.

        Returns:
            ConfigurableBuilder: The updated builder instance.
        """
        self._cli_args = args
        return self

    def build(self, model_id: str) -> T:
        """
        Builds and returns the object using the configured state and runtime parameters.

        Args:
            model_id (str): An identifier for the model to be built.

        Returns:
            T: An instance of the built object.
        """
        if self._cli_args is None:
            raise ValueError(
                "Builder is not configured. Call with_configuration() first."
            )
        return self._build(model_id)

    @abstractmethod
    def _build(self, model_id: str) -> T:
        raise NotImplementedError("Subclasses must implement the build method.")
