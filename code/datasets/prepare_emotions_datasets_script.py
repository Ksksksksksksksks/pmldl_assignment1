#%% md
# # Preparing Go-Emotions Dataset
#%% md
# I will use dataset with reddit comments and corresponding emotions from google - https://github.com/google-research/google-research/tree/master/goemotions
#%% md
# ### Imports
#%%
# !pip install iterative-stratification

#%%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import requests
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#%% md
# ### Create folders
#%%
# CURRENT_DIR = os.getcwd()
#
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
#
# FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "full_dataset")
# PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def main(data_dir, output_dir):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    #
    # print("FULL_DATA_DIR =", FULL_DATA_DIR)
    # print("PROCESSED_DIR =", PROCESSED_DIR)
    #%% md
    # ### Download datasets
    #%%
    urls = [
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
    ]

    for url in urls:
        filename = os.path.join(data_dir, url.split("/")[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename} ...")
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)
        else:
            print(f"{filename} already downloaded")
    #%% md
    # ### Read, see & concat
    #%%
    csv_files = [
        os.path.join(data_dir, f"goemotions_{i}.csv") for i in range(1, 4)
    ]

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    #%%
    # df.head()
    #%%
    # df.columns
    #%% md
    # #### Filter only clear emotions
    #%%
    df_filtered = df[df['example_very_unclear'] != 1]
    #%% md
    # ### Take only needed cols
    #%%
    selected_ems = ['anger', 'confusion','disgust',
                    'excitement', 'fear', 'joy',
                    'love', 'sadness', 'surprise','neutral']
    #%%

    #%%
    df_ems = df_filtered[['text', 'id'] + selected_ems]
    # df_ems.head()
    #%%
    df_ems = df_ems[df_ems[selected_ems].sum(axis=1) > 0]
    # df_ems.head()
    #%% md
    # ### Balance emotions
    #%%
    df_counts = df_ems[selected_ems].sum()
    min_count = df_counts.min()
    print(min_count)
    #%%
    df_balans = pd.concat(
        [df_ems[df_ems[emo] == 1].sample(min_count, random_state=42)
         for emo in selected_ems]
    ).drop_duplicates()
    df_balans.head()
    #%% md
    # ### Divide to train-val-split
    #%%
    # df_balans.size
    #%%
    X = df_balans[['text']]
    y = df_balans[selected_ems]

    # train + temp (80/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # temp -> val/test (50/50)
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    val_idx, test_idx = next(mskf.split(X_temp, y_temp))

    X_val = X_temp.iloc[val_idx]
    y_val = y_temp.iloc[val_idx]
    X_test = X_temp.iloc[test_idx]
    y_test = y_temp.iloc[test_idx]
    #%%
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    #%%
    X_train.assign(**y_train).to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    X_val.assign(**y_val).to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False)
    X_test.assign(**y_test).to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)
    #%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Go-Emotions dataset')
    parser.add_argument('--data_dir', type=str, default='/opt/airflow/data/full_dataset',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='/opt/airflow/data/processed',
                        help='Output directory')

    args = parser.parse_args()

    main(args.data_dir, args.output_dir)