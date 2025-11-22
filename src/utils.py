"""Utility functions"""


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Boolean value expected, got: {value}")
