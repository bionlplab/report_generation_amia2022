# Prior Knowledge Enhances Radiology Report Generation

[AMIA Informatics Summit 2022] [Prior Knowledge Enhances Radiology Report Generation](https://arxiv.org/abs/2201.03761)

Song Wang, Liyan Tang, Mingquan Lin, George Shih, Ying Ding, Yifan Peng

## Overview

The proposed model is defined in 'sentgcn.py'. 
To train a model, first train a gcnclassifier (multi-label classifier) as shown in run_gcnclassifier.sh, then train a sentgcn (report generation decoder) as shown in run_sentgcn.sh.

To train the gcnclassifier, you would need a DenseNet-121 model pretrained on ChexPert: model_ones_3epoch_densenet.tar.

The dataset splits, class keywords, embeddings and vocabs can be found in the data folder.

## Citation



## Acknowledgement


