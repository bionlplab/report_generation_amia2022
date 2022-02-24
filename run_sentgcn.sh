#! /usr/bin/env bash
python train_sentgcn.py --name sentgcn_30keywords_train401val2test3 --pretrained models/gcnclassifier_30keywords_train401val2test3_e100.pth --train-folds 401 --val-folds 2 --test-folds 3 --gpus 0 --batch-size 8 --decoder-lr 1e-6 --vocab-path /data/vocab.pkl
