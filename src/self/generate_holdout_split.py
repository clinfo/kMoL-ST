import os 
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np 
from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", required=True, type=str, 
        help="path to working directory. e.g., './data/chembl_all_10m_1u_09_01_12'"     
    )
    parser.add_argument(
        "-d", "--data", required=True, type=str, 
        help="input filepath. e.g. './data/chembl_v2_above_500_reduced.csv'"
    )
    parser.add_argument(
        "--stratified", required=False, type=strtobool, default='True'
    )
    parser.add_argument(
        "-i", "--iteration", required=True
    )
    parser.add_argument(
        "--seed", required=False, default=0
    )

    return parser.parse_args()

def generate_stratified_split(df, iter_id, test_size=0.2, output_location=None, seed=0):
    print("sample_size:", len(df))
    sample_idxs = np.array(df.index.tolist())
    target_ids = np.array(df["target_id"].tolist())
    train_idx, val_idx, _, _ = train_test_split(sample_idxs, target_ids, random_state=seed, stratify=target_ids, test_size=test_size)
            
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    split_dict = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist()
    }

    if output_location:
        opath = Path(output_location)
        json.dump(split_dict, open(opath / "iter{}_split.json".format(iter_id), "w"))

    return

def generate_split(df, iter_id, test_size=0.2, output_location=None, seed=0):
    print("sample_size:", len(df))
    sample_idxs = np.array(df.index.tolist())
    target_ids = np.array(df["target_id"].tolist())
    train_idx, val_idx, _, _ = train_test_split(sample_idxs, target_ids, random_state=seed, test_size=test_size)
            
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    split_dict = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist()
    }

    if output_location:
        opath = Path(output_location)
        json.dump(split_dict, open(opath / "iter{}_split.json".format(iter_id), "w"))

    return

def main():
    args = get_parser()
    path = args.path
    stratified = bool(args.stratified)
    iter_id = int(args.iteration)
    seed = int(args.seed)

    os.chdir(path)
    print(args.data)
    sample = pd.read_csv(args.data)
    
    if iter_id == 0:
        sample_idx = np.array(sample.index.tolist())
        target_ids = np.array(sample["target_id"].tolist())
        if stratified:
            train_val_idx, test_idx, _, _ = train_test_split(sample_idx, target_ids, random_state=seed, stratify=target_ids, test_size=0.2)
        else:
            train_val_idx, test_idx, _, _ = train_test_split(sample_idx, target_ids, random_state=seed, test_size=0.2)
        test_sample = sample.loc[test_idx.tolist()].reset_index(drop=True)
        test_dict = {
            "test": test_sample.index.tolist()
        }
        json.dump(test_dict, open("./split/test_split.json", "w"))
        train_val_sample = sample.loc[train_val_idx.tolist()].reset_index(drop=True)
        train_val_path = args.data.split('.csv')[0] + '_iter0.csv'
        train_val_sample.to_csv(train_val_path, index=False)
        if stratified:
            generate_stratified_split(train_val_sample, iter_id=iter_id, test_size=0.2, output_location='./split', seed=seed)
        else:
            generate_split(train_val_sample, iter_id=iter_id, test_size=0.2, output_location='./split', seed=seed)
        test_path = args.data.split('.csv')[0] + '_test.csv'
        test_sample.to_csv(test_path, index=False)

    else:
        if stratified:
            generate_stratified_split(sample, iter_id=iter_id, test_size=0.2, output_location='./split', seed=seed)
        else:
            generate_split(train_val_sample, iter_id=iter_id, test_size=0.2, output_location='./split', seed=seed)


if __name__ == "__main__":
    main()