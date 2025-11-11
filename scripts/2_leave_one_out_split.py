import os
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parsed_args = parser.parse_args()
    return parsed_args.data_dir


def main() -> None:
    data_dir = parse_args()
    data_dir_path = Path(data_dir)
    input_file = data_dir_path / "all_interactions.csv"
    output_dir = data_dir_path / "split"

    df = pd.read_csv(input_file)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_list, val_list, test_list = [], [], []

    for _, group in df.groupby("user_id", sort=False):
        train_list.append(group.iloc[:-2])
        val_list.append(group.iloc[[-2]])
        test_list.append(group.iloc[[-1]])

    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    train.drop(columns=["timestamp"], inplace=True)
    val.drop(columns=["timestamp"], inplace=True)
    test.drop(columns=["timestamp"], inplace=True)

    print(f"Train interactions: {len(train)}")
    print(f"Validation interactions: {len(val)}")
    print(f"Test interactions: {len(test)}")

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
