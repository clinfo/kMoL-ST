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
        help="input filepath. e.g. './data/chembl_31_gpcr.csv'"
    )
    parser.add_argument(
        "--target_name", required=False, type=str, default="t_1u", 
    )
    parser.add_argument(
        "--sample", required=False, type=str, default=None, 
        help="sampling data. e.g. './data/chembl_31_all.csv'"
    )
    parser.add_argument(
        "--test_path", required=False, type=str, default=None, 
        help="test path"
    )
    parser.add_argument(
        "-o", "--output", required=False, type=str, default=None, 
        help="output path"
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

def generate_random_negative(train_val_sample, sample_df=None, target_name="t_1u"):
    pair_ids = list(zip(train_val_sample['target_id'], train_val_sample['drug_id']))
    pair_set = set(pair_ids)
    target_ids = list(set(train_val_sample['target_id'].tolist()))
    if sample_df is not None:
        drug_ids = list(set(sample_df['drug_id'].tolist()))
    else:
        drug_ids = list(set(train_val_sample['drug_id'].tolist()))
    pooled_pair_set = set()

    
    for tar_id in target_ids:
        target_sample = train_val_sample[train_val_sample["target_id"]==tar_id]
        target_sample_size = max(2*len(target_sample.query(target_name+"==1")) - len(target_sample), 0)
        i = 0 
        while True:
            if i >= target_sample_size:
                break 
            drug_id = random.choice(drug_ids)
            query = (tar_id, drug_id)
            if not query in pair_set:
                if not query in pooled_pair_set:
                    pooled_pair_set.add(query)
                    i += 1 
                else:
                    pass 
            else:
                pass 
            
    pooled_pair_list = list(pooled_pair_set)
    pooled_target_list = [pair[0] for pair in pooled_pair_list]
    pooled_drug_list = [pair[1] for pair in pooled_pair_list]

    target_set = train_val_sample[['target_id', 'target_sequence']].drop_duplicates(keep='first').reset_index(drop=True).set_index('target_id')
    if sample_df is not None:
        drug_set = sample_df[['drug_id', 'smiles']].drop_duplicates(keep='first').reset_index(drop=True).set_index('drug_id')
    else:
        drug_set = train_val_sample[['drug_id', 'smiles']].drop_duplicates(keep='first').reset_index(drop=True).set_index('drug_id')
    nega_target = target_set.loc[pooled_target_list].reset_index(drop=False)
    nega_drug = drug_set.loc[pooled_drug_list].reset_index(drop=False)
    nega_df = pd.concat([nega_target, nega_drug], axis=1)
    nega_df[target_name] = 0 
    pos_neg = pd.concat([train_val_sample, nega_df], axis=0).reset_index(drop=True)
    
    return pos_neg

def main():
    args = get_parser()
    path = args.path 
    output_path = args.output

    os.chdir(path)
    sample = pd.read_csv(args.data)
    sample_idx = np.array(sample.index.tolist())
    target_ids = np.array(sample["target_id"].tolist())
    target_name=  args.target_name
    test_path = Path(args.test_path)
    print(test_path)
    if not test_path.exists():
        train_val_idx, test_idx, _, _ = train_test_split(sample_idx, target_ids, random_state=0, stratify=target_ids, test_size=0.2)
        test_sample = sample.loc[test_idx.tolist()].reset_index(drop=True)
        train_val_sample = sample.loc[train_val_idx.tolist()].reset_index(drop=True)
    else:
        train_val_sample = sample

    if output_path is not None:
        train_val_path = os.path.join(output_path, "data", args.data.split("/")[-1].split(".csv")[0].split("_iter0")[0] + "_train_val.csv")
        print(train_val_path)
    else:    
        train_val_path = args.data.split(".csv")[0].split("_iter0")[0] + "_train_val.csv"
    
    if args.sample is not None:
        sample_df = pd.read_csv(args.sample)
    else:
        sample_df = None
    train_val_sample = generate_random_negative(train_val_sample, sample_df, target_name)
    train_val_sample.to_csv(train_val_path, index=False)
    if output_path is not None:
        print(os.path.join(output_path, "split"))
        generate_split(train_val_sample, test_size=0.2, output_location=os.path.join(output_path, "split"))
    else:
        generate_split(train_val_sample, test_size=0.2, output_location="./split")
    
    if test_path.exists():
        pass
    else:
        test_sample.to_csv(test_path, index=False)

if __name__ == "__main__":
    main()
