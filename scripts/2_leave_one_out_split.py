import os
import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="data/processed/amazon-2014/all_interactions.csv",
        help="Path to the input interactions CSV file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data/processed/amazon-2014/split/",
        help="Directory to save the output split CSV files",
    )
    parsed_args = parser.parse_args()
    return parsed_args.input_file, parsed_args.output_dir


def main() -> None:
    input_file, output_dir = parse_args()

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

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
