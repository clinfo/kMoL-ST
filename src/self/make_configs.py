import os 
import json 
import copy
import argparse
from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", required=True, type=str, 
        help="path to working directory. e.g., './data/chembl_all_10m_1u_09_01_12'"     
    )
    parser.add_argument(
        "-i", "--iteration", required=True, type=int
    )
    parser.add_argument(
        "--noise", required=False, type=strtobool, default='False'
    )

    return parser.parse_args()

def make_train_config(conf, iter, noise=False):
    new_train_config = copy.deepcopy(conf)
    input = conf["loader"]["input_path"]
    new_train_config["loader"]["input_path"] = input.split('_iter0.csv')[0] + "_iter{}.csv".format(iter)
    split_path = conf["splitter"]["split_path"]
    new_split_path_list = split_path.split('/')[:-1]
    new_split_path_list.append('iter{}_split.json'.format(iter))
    new_train_config["splitter"]["split_path"] = '/'.join(new_split_path_list)
    output_path = conf["output_path"]
    new_output_path_list = output_path.split('/')[:-2]
    new_output_path_list.append('iter{}'.format(iter))
    new_output_path_list.append('train')
    new_train_config["output_path"] = '/'.join(new_output_path_list)

    if noise:
        new_train_config["model"]["protein_module"]["dropout"] += 0.01*iter
    new_train_config["model"]["ligand_module"]["dropout"] += 0.01*iter

    with open('./config/iter{}/train.json'.format(iter), "w") as f:
        json.dump(new_train_config, f, indent=4)

def make_sample_config(conf, iter, noise=None):
    for j in range(10):
        new_sample_config = copy.deepcopy(conf)
        input = conf["loader"]["input_path"]
        new_input = input.split("/")[:-2]
        new_input.append("iter{}".format(iter))
        new_input.append("sample{}.csv".format(j))
        new_sample_config["loader"]["input_path"] = "/".join(new_input)
        output = conf["output_path"]
        new_output = output.split("/")
        new_output[-3] = "iter{}".format(iter)
        new_output[-1] = "sample{}".format(j)
        new_sample_config["output_path"] = "/".join(new_output)
        checkpoint = conf["checkpoint_path"]
        new_checkpoint = checkpoint.split("/")
        new_checkpoint[-3] = "iter{}".format(iter-1)
        new_sample_config["checkpoint_path"] = "/".join(new_checkpoint)

        if noise:
            new_sample_config["model"]["protein_module"]["dropout"] += 0.01*(iter-1)
            new_sample_config["model"]["ligand_module"]["dropout"] += 0.01*(iter-1)
        
        with open('./config/iter{}/sample_predict_{}.json'.format(iter, j), "w") as f:
            json.dump(new_sample_config, f, indent=4)

def make_test_config(conf, iter, noise=None, name="test"):
    new_test_config = copy.deepcopy(conf)
    output = conf["output_path"]
    new_output = output.split("/")
    new_output[-2] = "iter{}".format(iter)
    new_test_config["output_path"] = "/".join(new_output)
    checkpoint = conf["checkpoint_path"]
    new_checkpoint = checkpoint.split("/")
    new_checkpoint[-3] = "iter{}".format(iter)
    new_test_config["checkpoint_path"] = "/".join(new_checkpoint)

    if noise:
            new_test_config["model"]["protein_module"]["dropout"] += 0.01*iter
            new_test_config["model"]["ligand_module"]["dropout"] += 0.01*iter
    
    with open("./config/iter{}/{}_predict.json".format(iter, name), "w") as f:
            json.dump(new_test_config, f, indent=4)


def main():
    args = get_parser()
    path = args.path
    iteration = args.iteration
    noise = bool(args.noise)

    os.chdir(path)
    with open("./config/iter0/train.json", "r") as f:
        train_config = json.load(f)
    
    with open("./config/iter1/sample_predict_0.json", "r") as f:
        sample_config = json.load(f)

    with open("./config/iter0/test_predict.json", "r") as f:
        test_config = json.load(f)
    
    if os.path.isfile("./config/iter0/bioprint_predict.json"):
        with open("./config/iter0/bioprint_predict.json", "r") as f:
            bioprint_config = json.load(f)
    
    if os.path.isfile("./config/iter0/bindingdb_all_predict.json"):
        with open("./config/iter0/bindingdb_all_predict.json", "r") as f:
            bindingdb_config = json.load(f)
    
    if os.path.isfile("./config/iter0/davis_predict.json"):
        with open("./config/iter0/davis_predict.json", "r") as f:
            davis_config = json.load(f)

    for i in range(1, iteration+1):
        os.makedirs("./config/iter{}".format(i), exist_ok=True)
        make_train_config(train_config, i, noise)
        make_sample_config(sample_config, i, noise)
        make_test_config(test_config, i, noise)
        if os.path.isfile("./config/iter0/bioprint_predict.json"):
            make_test_config(bioprint_config, i, noise, "bioprint")
        if os.path.isfile("./config/iter0/bindingdb_all_predict.json"):
            make_test_config(bindingdb_config, i, noise, "bindingdb_all")
        if os.path.isfile("./config/iter0/davis_predict.json"):
            make_test_config(davis_config, i, noise, "davis")

if __name__ == "__main__":
    main()
