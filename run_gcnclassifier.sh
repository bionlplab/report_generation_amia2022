#! /usr/bin/env bash

python /home/sw37643/ReportGenerationMeetsGraph/train_gcnclassifier.py --name gcnclassifier_30keywords_t401v2t3 --pretrained /home/sw37643/ReportGenerationMeetsGraph/models/pretrained/model_ones_3epoch_densenet.tar --dataset-dir /home/sw37643/ReportGenerationMeetsGraph/data/NLMCXR_png --train-folds 401 --val-folds 2 --test-folds 3 --lr 1e-6 --batch-size 8 --gpus 0 --num-epochs 150
