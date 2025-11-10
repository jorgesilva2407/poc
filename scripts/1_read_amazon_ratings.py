import os
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = ArgumentParser(description="Process Amazon Ratings Dataset")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed files",
    )
    parser.add_argument(
        "-k",
        "--k-core",
        type=int,
        default=5,
        help="Core value for iterative filtering (e.g., 5 for 5-core).",
    )
    parsed_args = parser.parse_args()
    return parsed_args.input_file, parsed_args.output_dir, parsed_args.k_core


def filter_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Iteratively filter df until all users and items have k interactions.

    Args:
        df: The interactions dataframe.
        k: The minimum number of interactions for users and items.
    """
    print(f"Applying {k}-core filter...")
    while True:
        initial_interactions = len(df)

        # 1. Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]

        # 2. Filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]

        final_interactions = len(df)

        if initial_interactions == final_interactions:
            # No interactions were removed in this pass, so we are done
            print("k-core filter applied.")
            break
        else:
            print(
                f"  - Iteration removed {initial_interactions - final_interactions} interactions..."
            )

    return df


def process_dataset(jsonl_file: str, output_dir: str, k_core: int = 5):
    """
    Process a single dataset: read, filter, index, and save.

    Args:
        jsonl_file: Path to the input JSONL file
        output_dir: Directory where processed files will be saved
        k_core: The core value for iterative filtering.
    """
    # Read ratings
    df = pd.read_json(jsonl_file, lines=True)
    df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
    df.columns = ["user_id", "item_id", "rating", "timestamp"]

    # Convert to implicit
    df = df[df["rating"] >= 4]
    df.drop(columns=["rating"], inplace=True)

    # Apply iterative k-core filter
    df = filter_k_core(df, k=k_core)

    # Index columns (after filtering)
    user_encoder = LabelEncoder()
    df["user_id"] = user_encoder.fit_transform(df["user_id"])

    item_encoder = LabelEncoder()
    df["item_id"] = item_encoder.fit_transform(df["item_id"])

    # Save outputs
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(output_path / "all_interactions.csv", index=False)

    with open(output_path / "user_encoder.pkl", "wb") as f:
        pkl.dump(user_encoder, f)

    with open(output_path / "item_encoder.pkl", "wb") as f:
        pkl.dump(item_encoder, f)

    print(f"Processed {jsonl_file}")
    print(f"  - Total interactions: {len(df)}")
    print(f"  - Unique users: {df['user_id'].nunique()}")
    print(f"  - Unique items: {df['item_id'].nunique()}")
    print(f"  - Saved to: {output_dir}")


def main() -> None:
    input_file, output_dir, k_core = parse_args()

    process_dataset(
        jsonl_file=input_file,
        output_dir=output_dir,
        k_core=k_core,
    )


if __name__ == "__main__":
    main()
