from copy import deepcopy
import pandas as pd 
import numpy as np 
import argparse
import random 
import json 
import os 
from tqdm import tqdm
import sys 
from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", required=True, type=str
    )

    parser.add_argument(
        "-d", "--data", required=True
    )

    parser.add_argument(
        "-t", "--target_name", required=False, default="t_1u"
    )

    parser.add_argument(
        "--external", required=False, default=None
    )

    parser.add_argument(
        "-n", "--num", required=True
    )

    parser.add_argument(
        "-i", "--iteration", required=True
    )
    parser.add_argument(
        "--bin_aug", required=False, type=strtobool, default='False'
    )
    parser.add_argument(
        "--only_negative", required=False, type=strtobool, default='False'
    )

    return parser.parse_args()


def main():
    print("sampling start")
    args = get_parser()
    path = args.path 
    data = args.data
    target_name = args.target_name
    iter = int(args.iteration)
    num = int(args.num)
    external = args.external
    bin_aug =  bool(args.bin_aug)
    only_negative = bool(args.only_negative)

    print(path)
    os.chdir(path)

    print(data)
    sample = pd.read_csv(data)
    print(sample.shape)
    
    pair_ids = list(zip(sample["target_id"], sample["drug_id"]))
    print("pair_ids", len(pair_ids))
    pair_set = set(pair_ids)
    print("pair_set", len(pair_set))

    if external is not None:
        external_df = pd.read_csv(external)
        print("external_df:", external_df.shape)
    
    target_ids = list(set(sample["target_id"].tolist()))

    adding_nums = []
    target_totals = []
    target_pos = []
    for tar_id in tqdm(target_ids):
        target_sample = sample[sample["target_id"]==tar_id]
        tar_total = len(target_sample)
        target_totals.append(tar_total)
        tar_pos = target_sample[target_name].sum()
        target_pos.append(tar_pos)
        if only_negative:
            adding_num = max(0, 2*tar_pos-tar_total)
        else:
            adding_num = abs(2*tar_pos - tar_total)
        adding_nums.append(adding_num)
    print("adding nums:",  np.sum(adding_nums))

    if np.sum(adding_nums) == 0:
        if not bin_aug:
            print("=== Finished! ===")
        else:
            adding_nums = np.array(max(target_totals)-2*np.array(target_pos)).tolist()

    if external is not None:
        drug_ids = list(set(external_df["drug_id"].tolist()))
    else:
        drug_ids = list(set(sample["drug_id"].tolist()))
    print("drug_ids:", len(drug_ids))

    possible_num = sum((np.array(adding_nums)>0)*(len(drug_ids)-np.array(target_totals)))
    num = min(possible_num, num) # 221210 modified
    print("num: {}".format(num))
    
    pooled_pair_set = set()

    while True:
        if np.sum(adding_nums)==0:
            break
        target_id = random.choices(target_ids, k=1, weights=adding_nums)[0]
        drug_id = random.choice(drug_ids)
        query = (target_id, drug_id)
        if not query in pair_set:
            if not query in pooled_pair_set:
                pooled_pair_set.add(query)
            else:
                pass 
        else:
            pass

        if len(pooled_pair_set) == num:
            break
    
    pooled_pair_list = list(pooled_pair_set)
    pooled_target_list = [pair[0] for pair in pooled_pair_list]
    pooled_drug_list = [pair[1] for pair  in pooled_pair_list]

    target_set = sample[["target_id", "target_sequence"]].drop_duplicates(keep="first").set_index("target_id")
    if external is not None:
        drug_set = external_df[["drug_id", "smiles"]].drop_duplicates(keep="first").reset_index(drop=True).set_index("drug_id")
    else:
        drug_set = sample[["drug_id", "smiles"]].drop_duplicates(keep="first").reset_index(drop=True).set_index("drug_id")

    pooled_target = target_set.loc[pooled_target_list].reset_index(drop=False)
    pooled_drug = drug_set.loc[pooled_drug_list].reset_index(drop=False)
    pooled_df = pd.concat([pooled_target, pooled_drug], axis=1)

    pooled_df[target_name] = 0 

    os.makedirs("./data/iter{}".format(iter+1), exist_ok=True)
    for i in range(10):
        pool = pooled_df.iloc[i*num//10:(i+1)*num//10]
        pool.to_csv("./data/iter{}/sample{}.csv".format(iter+1, i), index=False)


if __name__ == "__main__":
    main()        

    

