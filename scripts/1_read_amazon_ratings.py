import os
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_ratings(ratings_file):
    df = pd.read_json(ratings_file, lines=True)
    df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    df = df[df["rating"] >= 4]
    df.drop(columns=["rating"], inplace=True)
    return df


def merge_ratings(beauty_ratings, clothing_ratings):
    merged_ratings = pd.concat([beauty_ratings, clothing_ratings], ignore_index=True)
    return merged_ratings


def filter_ratings(df, min_user_ratings=3):
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    filtered_df = df[df["user_id"].isin(valid_users)]
    return filtered_df


def index_column(df, column_name):
    encoder = LabelEncoder()
    df[column_name] = encoder.fit_transform(df[column_name])
    return df, encoder


def save_ratings(ratings, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ratings.to_csv(output_file, index=False)


def save_encoder(encoder, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pkl.dump(encoder, f)


def main() -> None:
    beauty_ratings = read_ratings("./data/raw/reviews_Beauty_5.jsonl")
    clothing_ratings = read_ratings(
        "./data/raw/reviews_Clothing_Shoes_and_Jewelry_5.jsonl"
    )
    merged_ratings = merge_ratings(beauty_ratings, clothing_ratings)
    merged_ratings = filter_ratings(merged_ratings, min_user_ratings=3)
    merged_ratings, user_encoder = index_column(merged_ratings, "user_id")
    merged_ratings, item_encoder = index_column(merged_ratings, "item_id")
    save_ratings(merged_ratings, "./data/processed/amazon-2014/all_interactions.csv")
    save_encoder(user_encoder, "./data/processed/amazon-2014/user_encoder.pkl")
    save_encoder(item_encoder, "./data/processed/amazon-2014/item_encoder.pkl")


if __name__ == "__main__":
    main()
