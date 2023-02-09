#!/bin/sh


last=$(date "+%s")

iteration=9
path="./data/self_training_kinase/cv1/kinase_self_training" # path from kmol

#source ~/.bashrc
#conda activate kmol

cd ../../../../

python src/self/make_configs.py -p ${path} -i ${iteration}

for i in `seq 1 ${iteration}`
do 
    echo $i
    if [ $i -gt 1 ]; then
        for j in `seq 0 9`
        do
            echo $j
            kmol predict "${path}/config/iter${i}/sample_predict_${j}.json"
        done
        python src/self/sample_selection.py --path ${path} -d "./data/chembl_31_kinase_iter`expr $i - 1`.csv" -t "t_10u" -i $i --prob_inf 0.20 --prob_sup 0.50 --pos_prob 0.99 --pos_rank 0.01
    fi
    if [ $i -eq 0 ]; then
        :
    else
        python src/self/generate_holdout_split.py -p ${path} -d "./data/chembl_31_kinase_iter${i}.csv" -i $i
    fi
    kmol train "${path}/config/iter${i}/train.json"
    kmol predict "${path}/config/iter${i}/test_predict.json"
    kmol predict "${path}/config/iter${i}/bioprint_predict.json"
    kmol predict "${path}/config/iter${i}/bindingdb_all_predict.json"
    kmol predict "${path}/config/iter${i}/davis_predict.json"
    python src/self/sampling.py -p ${path} -d "./data/chembl_31_kinase_iter${i}.csv" -t "t_10u" --external "../../data/chembl_31_all.csv" -n 500000 -i $i
done

current=$(date "+%s")
s=$((current - last))
echo "Elsaped Time: ${s} sec."