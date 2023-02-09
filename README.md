# kMoL-ST

## Installation

Dependencies can be installed with conda:
```bash
conda env create -f environment.yml
conda activate kmol
bash install.sh
```

## Usage of kMoL

Please refer to the original repository for the usage of kMoL (https://github.com/elix-tech/kmol). 

## Directory structure

```
.
├── data
|   ├── self_training_kinase             : data for kinase families
|        ├──cv1
|            ├──kinase_self_training
|                ├── config/             : configuration file
|                ├── data/               : dataset (csv)
|                ├── split/              : json files for split
|                └── run.sh              : script for running self-training
├── docker/                              : 
├── src                                  : source code
│   ├── kmol/                            : modules for kmol
│   ├── mila/                            : modules for federated learning
│   └── self                             : scripts for self-training
├── LICENSE                              : LICENSE file
└── README.md                            : this file
