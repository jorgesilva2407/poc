import os
import json
import pickle as pkl
from pathlib import Path
from typing import Literal
from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DataTypes = Literal["boolean", "numerical", "categorical"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--user-data-path",
        type=str,
        required=True,
        help="Path to the user data file (e.g., u.user)",
    )
    parser.add_argument(
        "--item-data-path",
        type=str,
        required=True,
        help="Path to the item data file (e.g., u.item)",
    )
    parser.add_argument(
        "--user-encoder-path",
        type=str,
        required=True,
        help="Path to load the user label encoder",
    )
    parser.add_argument(
        "--item-encoder-path",
        type=str,
        required=True,
        help="Path to load the item label encoder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the processed features",
    )
    args = parser.parse_args()
    # Return all arguments including encoders
    return (
        args.user_data_path,
        args.item_data_path,
        args.user_encoder_path,
        args.item_encoder_path,
        args.output_path,
    )


def process_user_data(
    user_data_path, output_path, user_encoder: LabelEncoder
) -> dict[str, DataTypes]:
    """Reads u.user, encodes features, and saves users.csv + encoders."""

    # 1. Load Data
    # Format: user id | age | gender | occupation | zip code
    df = pd.read_csv(
        user_data_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"zip_code": str},
    )

    # 2. Filter and Encode User ID
    known_users = set(user_encoder.classes_)
    df = df[df["user_id"].isin(known_users)].copy()
    df["user_id"] = user_encoder.transform(df["user_id"])

    # 3. Boolean Gender (M=1, F=0)
    df["gender_bool"] = df["gender"].map({"M": 1, "F": 0}).astype(int)

    # 4. Normalize Age
    scaler_age = MinMaxScaler()
    df["norm_age"] = scaler_age.fit_transform(df[["age"]])

    # 5. Occupation (Categorical -> Int Index)
    occ_encoder = LabelEncoder()
    df["occupation_idx"] = occ_encoder.fit_transform(df["occupation"])

    # 6. Zip-code (Categorical -> Int Index)
    zip_encoder = LabelEncoder()
    df["zip_idx"] = zip_encoder.fit_transform(df["zip_code"])

    # 7. Select Final Columns
    final_df = df[["user_id", "gender_bool", "norm_age", "occupation_idx", "zip_idx"]]

    # 8. Save CSV
    os.makedirs(output_path, exist_ok=True)
    final_df.to_csv(output_path / "users.csv", index=False)

    # 9. Save Encoders in subfolder
    encoders_dir = output_path / "encoders" / "user"
    os.makedirs(encoders_dir, exist_ok=True)

    with open(encoders_dir / "occupation_encoder.pkl", "wb") as f:
        pkl.dump(occ_encoder, f)
    with open(encoders_dir / "zip_encoder.pkl", "wb") as f:
        pkl.dump(zip_encoder, f)
    with open(encoders_dir / "age_scaler.pkl", "wb") as f:
        pkl.dump(scaler_age, f)

    # 10. Return Metadata
    return {
        "user_id": "categorical",
        "gender_bool": "boolean",
        "norm_age": "numerical",
        "occupation_idx": "categorical",
        "zip_idx": "categorical",
    }


def process_item_data(
    item_data_path, output_path, item_encoder: LabelEncoder
) -> dict[str, DataTypes]:
    """Reads u.item, encodes features, and saves items.csv + encoders."""

    # 1. Load Data
    # Format: movie id | movie title | release date | video release date |
    # IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy |
    # Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical |
    # Mystery | Romance | Sci-Fi | Thriller | War | Western
    genre_columns = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    df = pd.read_csv(
        item_data_path,
        sep="|",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
        ]
        + genre_columns,
        encoding="latin-1",
    )

    # 2. Filter and Encode Item ID
    known_items = set(item_encoder.classes_)
    df = df[df["item_id"].isin(known_items)].copy()
    df["item_id"] = item_encoder.transform(df["item_id"])

    # 3. Extract and Normalize Year
    # Regex to find "(1995)" at end of string
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)
    df["year"] = df["year"].fillna(df["year"].median())

    scaler_year = MinMaxScaler()
    df["norm_year"] = scaler_year.fit_transform(df[["year"]])

    # 4. Rename genre columns to match format
    genre_rename = {g: f"genre_{g}" for g in genre_columns}
    df = df.rename(columns=genre_rename)
    genre_column_names = [f"genre_{g}" for g in genre_columns]

    # 5. Combine
    final_df = pd.concat([df[["item_id", "norm_year"]], df[genre_column_names]], axis=1)

    # 6. Save CSV
    os.makedirs(output_path, exist_ok=True)
    final_df.to_csv(output_path / "items.csv", index=False)

    # 7. Save Encoders in subfolder
    encoders_dir = output_path / "encoders" / "item"
    os.makedirs(encoders_dir, exist_ok=True)

    with open(encoders_dir / "year_scaler.pkl", "wb") as f:
        pkl.dump(scaler_year, f)

    # 8. Return Metadata
    metadata = {
        "item_id": "categorical",
        "norm_year": "numerical",
    }
    for col in genre_column_names:
        metadata[col] = "boolean"

    return metadata


def main():
    """Main function to process ML-100K user and item features."""
    # 1. Parse Args
    (
        user_data_path,
        item_data_path,
        user_encoder_path,
        item_encoder_path,
        output_path,
    ) = parse_args()

    # Convert to Path objects
    user_data_path = Path(user_data_path)
    item_data_path = Path(item_data_path)
    output_path = Path(output_path)

    # 2. Load existing ID Encoders
    with open(user_encoder_path, "rb") as f:
        user_encoder = pkl.load(f)

    with open(item_encoder_path, "rb") as f:
        item_encoder = pkl.load(f)

    # 3. Process Data
    user_features = process_user_data(user_data_path, output_path, user_encoder)

    item_features = process_item_data(item_data_path, output_path, item_encoder)

    # 4. Save Metadata
    features_metadata = {
        "user": user_features,
        "item": item_features,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(features_metadata, f, indent=4)

    print(json.dumps(features_metadata, indent=4))


if __name__ == "__main__":
    main()
