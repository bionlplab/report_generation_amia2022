# Prior Knowledge Enhances Radiology Report Generation

[AMIA Informatics Summit 2022] [Prior Knowledge Enhances Radiology Report Generation](https://arxiv.org/abs/2201.03761)

Song Wang, Liyan Tang, Mingquan Lin, George Shih, Ying Ding, Yifan Peng

## Overview

The proposed model is defined in 'sentgcn.py'. 
To train a model, first train a gcnclassifier (multi-label classifier) as shown in run_gcnclassifier.sh, then train a sentgcn (report generation decoder) as shown in run_sentgcn.sh.

To train the gcnclassifier, you would need a DenseNet-121 model pretrained on ChexPert: model_ones_3epoch_densenet.tar.

The dataset splits, class keywords, embeddings and vocabs can be found in the data folder.

## Usage


## Citation
If you find our work helpful, please cite:
```
@misc{wang2022prior,
      title={Prior Knowledge Enhances Radiology Report Generation}, 
      author={Song Wang and Liyan Tang and Mingquan Lin and George Shih and Ying Ding and Yifan Peng},
      year={2022},
      eprint={2201.03761},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement
This work is supported by Amazon Machine Learning Research Award 2020. It was also supported by the National Library of Medicine under Award No. 4R00LM013001. 

We also appreciate the authors of [When Radiology Report Generation Meets Knowledge Graph](https://arxiv.org/abs/2002.08277) for providing the codes.

