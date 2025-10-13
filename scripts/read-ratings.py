import json
import pandas as pd

def read_ratings(ratings_file):
    df = pd.read_json(ratings_file, lines=True)
    df = df[['reviewerID', 'asin', 'overall']]
    df.columns = ['user_id', 'item_id', 'rating']
    return df

def merge_ratings(beauty_ratings, clothing_ratings):
    merged_ratings = pd.concat([beauty_ratings, clothing_ratings], ignore_index=True)
    return merged_ratings

def index_column(df, column_name):
    unique_values = df[column_name].unique()
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(value_to_index)
    return df, value_to_index

def save_ratings(ratings, output_file):
    ratings.to_csv(output_file, index=False)

def save_index_mapping(mapping, output_file):
    with open(output_file, 'w') as f:
        json.dump(mapping, f)

def main() -> None:
    beauty_ratings = read_ratings('./data/raw/reviews_Beauty_5.jsonl')
    clothing_ratings = read_ratings('./data/raw/reviews_Clothing_Shoes_and_Jewelry_5.jsonl')
    merged_ratings = merge_ratings(beauty_ratings, clothing_ratings)
    merged_ratings, user_id_index = index_column(merged_ratings, 'user_id')
    merged_ratings, item_id_index = index_column(merged_ratings, 'item_id')
    save_ratings(merged_ratings, './data/processed/merged_ratings.csv')
    save_index_mapping(user_id_index, './data/processed/user_id_index.json')
    save_index_mapping(item_id_index, './data/processed/item_id_index.json')

if __name__ == '__main__':
    main()
