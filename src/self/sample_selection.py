import pickle 
import numpy as np 
import pandas as pd 
import argparse
import os 
import sys 
from distutils.util import strtobool


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", required=True, type=str
    )

    parser.add_argument(
        "-d", "--data", required=True, type=str
    )

    parser.add_argument(
        "-t", "--target_name", required=False, default="t_1u"
    )

    parser.add_argument(
        "-i", "--iteration", required=True
    )

    parser.add_argument(
        "--prob_inf", required=False, default=0.0
    )

    parser.add_argument(
        "--prob_sup", required=False, default=1.0
    )

    parser.add_argument(
        "--rank_inf", required=False, default=0.0
    )

    parser.add_argument(
        "--rank_sup", required=False, default=1.0
    )

    parser.add_argument(
        "--mc_inf", required=False, default=0.0
    )

    parser.add_argument(
        "--pos_prob", required=True, 
    )

    parser.add_argument(
        "-r", "--pos_rank", required=True
    )
    
    parser.add_argument(
        "-v", "--pos_var", required=False, default=None
    )

    return parser.parse_args()

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def main():
    args = get_parser()
    path = args.path 
    target_name = args.target_name
    data = args.data
    iter = int(args.iteration)
    prob_inf = float(args.prob_inf)
    prob_sup = float(args.prob_sup)
    rank_inf = float(args.rank_inf)
    rank_sup = float(args.rank_sup)
    mc_inf = float(args.mc_inf)
    pos_prob = float(args.pos_prob)
    pos_rank = float(args.pos_rank)
    pos_var = float(args.pos_var) if args.pos_var is not None else None
    os.chdir(path)

    samples = []
    for j in range(10):
        prediction = pd.read_csv("result/iter{}/sample_predict/sample{}/predictions.csv".format(iter, j))
        prediction = prediction.sort_values(by="id") # 22.12.05 koyama
        prediction = prediction.drop("id", axis=1)
        prediction = prediction.reset_index(drop=True)
        added = pd.read_csv("data/iter{}/sample{}.csv".format(iter, j))
        added = added.drop(target_name, axis=1)
        sample = pd.concat([added, prediction], axis=1)
        samples.append(sample)
    samples = pd.concat(samples, axis=0).reset_index(drop=True)
    samples["prob"] = sigmoid(samples[target_name])

    original_sample = pd.read_csv(data)

    target_ids = list(set(original_sample.target_id.tolist()))
    neg_aug = []
    pos_aug = []
    for i in target_ids:
        total_sample = len(original_sample[original_sample["target_id"]==i])
        pos_sample = int(original_sample[original_sample["target_id"]==i][target_name].sum())
        if pos_sample / total_sample > 0.5:
            neg_added = samples[samples["target_id"]==i].sort_values(by="prob", ascending=True)
            neg_added = neg_added.iloc[int(len(neg_added)*rank_inf):int(len(neg_added)*rank_sup)]
            neg_added = neg_added[(neg_added["prob"]<=prob_sup) & (neg_added["prob"]>=prob_inf)]
            neg_added = neg_added[neg_added[target_name + "_logits_var"]>=mc_inf]
            random_index = np.random.permutation(neg_added.index)
            neg_added = random_index[:int(2*pos_sample-total_sample)]
            neg_aug += list(neg_added)

        elif pos_sample / total_sample < 0.5:
            pos_added = samples[samples["target_id"]==i].sort_values(by="prob", ascending=False)
            pos_added = pos_added.iloc[:int(len(pos_added)*pos_rank)]
            pos_added = pos_added[pos_added["prob"]>=pos_prob]
            if pos_var is not None:
                pos_added = pos_added[pos_added[target_name + "_logits_var"]<=var]
            pos_added = pos_added.iloc[:int(total_sample-2*pos_sample)].index
            pos_aug += list(pos_added)
                

    neg_sample_aug = samples.iloc[neg_aug]
    pos_sample_aug = samples.iloc[pos_aug]

    columns = list(samples.columns)
    for i in ['{}_ground_truth'.format(target_name), '{}_logits_var'.format(target_name), 'prob']:
        if i in columns:
            columns.remove(i)

    contain_pchembl = "pchembl_value" in columns
    
    columns.remove(target_name)

    neg_sample_reduced = neg_sample_aug[columns]
    pos_sample_reduced = pos_sample_aug[columns]

    if contain_pchembl:
        neg_sample_reduced["pchembl_value"] = 0 
        pos_sample_reduced["pchembl_value"] = 0
        
    neg_sample_reduced[target_name] = 0   
    pos_sample_reduced[target_name] = 1 

    if not contain_pchembl:
        columns.append("pchembl_value")
    
    columns.append(target_name)

    sample_aug  = pd.concat([original_sample.loc[:,columns], neg_sample_reduced, pos_sample_reduced],  axis=0)
    sample_aug = sample_aug.sort_values(by="target_id").reset_index(drop=True)
    data_path_root  = data.split("_iter{}.csv".format(iter-1))[0]
    sample_aug.to_csv(data_path_root + "_iter{}.csv".format(iter), index=False) 

if __name__ == "__main__":
    main()