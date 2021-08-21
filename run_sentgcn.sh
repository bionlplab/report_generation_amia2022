#! /usr/bin/env bash
python /home/sw37643/ReportGenerationMeetsGraph/train_sentgcn.py --name sentgcn_30keywords_t401v2t3_ctx --pretrained /home/sw37643/ReportGenerationMeetsGraph/models/gcnclassifier_30keywords_t401v2t3_e100.pth --train-folds 401 --val-folds 2 --test-folds 3 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /home/sw37643/ReportGenerationMeetsGraph/data/vocab.pkl
