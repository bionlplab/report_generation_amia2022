#! /usr/bin/env bash

python train_gcnclassifier.py --name gcnclassifier_30keywords_train401val2test3 --pretrained /models/pretrained/model_ones_3epoch_densenet.tar --dataset-dir /data/openi --train-folds 401 --val-folds 2 --test-folds 3 --lr 1e-6 --batch-size 8 --gpus 0 --num-epochs 150