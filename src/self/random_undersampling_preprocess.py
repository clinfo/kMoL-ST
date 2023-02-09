import os
import pandas as pd 
import json 
import random 
import argparse
from pathlib import Path
import numpy as np 
from sklearn.model_selection import train_test_split 


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", required=True, type=str, 
        help="path to working directory. e.g., './data/random_negative'"
    )
    parser.add_argument(
        "-d", "--data", required=True, type=str, 
        help="input filepath. e.g. './data/chembl_v2_above_500.csv'"
    )
    return parser.parse_args()

def generate_split(df, test_size=0.2, output_location=None):
    print("sample_size:", len(df))
    sample_idxs = np.array(df.index.tolist())
    target_ids = np.array(df["target_id"].tolist())
    train_idx, val_idx, _, _ = train_test_split(sample_idxs, target_ids, random_state=0, stratify=target_ids, test_size=test_size)
            
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    split_dict = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist()
    }

    if output_location:
        opath = Path(output_location)
        json.dump(split_dict, open(opath / "split.json", "w"))

    return

def generate_random_undersample(train_val_sample):
    pos_train_valid = train_val_sample[train_val_sample['t_1u']==1]
    neg_train_valid = train_val_sample[train_val_sample["t_1u"]==0]

    if len(pos_train_valid) > len(neg_train_valid):
        pos_train_valid = pos_train_valid.sample(n=len(neg_train_valid), random_state=0)
    elif len(pos_train_valid) < len(neg_train_valid):
        neg_train_valid = neg_train_valid.sample(n=len(neg_train_valid), random_state=0)

    pos_neg = pd.concat([pos_train_valid, neg_train_valid], axis=0).reset_index(drop=True)
    
    return pos_neg

def main():
    args = get_parser()
    path = args.path 

    os.chdir(path)

    sample = pd.read_csv(args.data)
    sample_idx = np.array(sample.index.tolist())
    target_ids = np.array(sample["target_id"].tolist())
    train_val_idx, test_idx, _, _ = train_test_split(sample_idx, target_ids, random_state=0, stratify=target_ids, test_size=0.2)
    test_sample = sample.loc[test_idx.tolist()].reset_index(drop=True)
    test_path = Path(args.data.split(".csv")[0].split("_iter0")[0] + "_test.csv")
    if not test_path.exists():
        train_val_idx, test_idx, _, _ = train_test_split(sample_idx, target_ids, random_state=0, stratify=target_ids, test_size=0.2)
        test_sample = sample.loc[test_idx.tolist()].reset_index(drop=True)
        test_sample.to_csv(test_path, index=False)
        train_val_sample = sample.loc[train_val_idx.tolist()].reset_index(drop=True)
    else:
        train_val_sample = sample
    
    train_val_file = args.data.split("/")[-1].split("_iter0")[0].split(".csv")[0] + "_train_val.csv"
    train_val_path = os.path.join("./data", train_val_file)
    train_val_sample = generate_random_undersample(train_val_sample)
    train_val_sample.to_csv(train_val_path, index=False)
    generate_split(train_val_sample, test_size=0.2, output_location="./split")

if __name__ == "__main__":
    main()
