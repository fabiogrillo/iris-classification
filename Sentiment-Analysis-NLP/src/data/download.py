# This file is intended to download the IMDB dataset if not already present.
from datasets import load_dataset
import pandas as pd

def download_imdb():
    # Load the IMDB dataset from Hugging Face datasets library
    dataset = load_dataset("imdb")

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Save
    train_df.to_csv('data/raw/train.csv', index=False)
    test_df.to_csv('data/raw/test.csv', index=False)

    print(f"Downloaded IMDB dataset with {len(train_df)} training samples and {len(test_df)} test samples.")

if __name__ == "__main__":
    download_imdb()
