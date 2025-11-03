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
