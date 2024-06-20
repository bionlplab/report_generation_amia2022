# Prior Knowledge Enhances Radiology Report Generation

## Overview

The proposed model is defined in `sentgcn.py`. 

1. To train a model, first train a gcnclassifier (multi-label classifier) model as shown in `run_gcnclassifier.sh`;
2. Then train a sentgcn (report generation decoder) model as shown in `run_sentgcn.sh` using the best gcnclassifier model from last step.

To train the gcnclassifier, you would need a DenseNet-121 model pretrained on ChexPert `model_ones_3epoch_densenet.tar`. The dataset splits, class keywords, embeddings and vocabs can be found in the `/data` folder. Image paths can be modified inside `/data/fold{}.txt` files.


## Citation
If you find our work helpful, please cite:

Wang S, Tang L, Lin M, Shih G, Ding Y, Peng Y. Prior Knowledge Enhances Radiology Report Generation. AMIA Jt Summits Transl Sci Proc. 2022 May 23;2022:486-495. PMID: 35854760; PMCID: PMC9285179.

## Acknowledgement
This work is supported by Amazon Machine Learning Research Award 2020. It was also supported by the National Library of Medicine under Award No. 4R00LM013001. 

Special thanks to the authors of [When Radiology Report Generation Meets Knowledge Graph](https://arxiv.org/abs/2002.08277) for providing the codes.

