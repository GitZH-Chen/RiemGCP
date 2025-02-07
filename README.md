[<img src="https://img.shields.io/badge/arXiv-2407.10484-b31b1b"></img>](https://arxiv.org/abs/2407.10484)
[<img src="https://img.shields.io/badge/OpenReview|forum-q1t0Lmvhty-8c1b13"></img>](https://openreview.net/forum?id=q1t0Lmvhty)
[<img src="https://img.shields.io/badge/OpenReview|pdf-q1t0Lmvhty-8c1b13"></img>](https://openreview.net/pdf?id=q1t0Lmvhty)

# Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry

## Introduction
This is the official code for our ICLR 2025 publication: *Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry*. 
For the first time, this work reveals that matrix functions in GCP implicitly respect Riemannian classifiers and explains the rationale behind using Euclidean classifiers on SPD matrices after applying the matrix power.

If you find this project helpful, please consider citing us as follows:
```bib
@inproceedings{chen2025riemgcp,
  title={Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of {Riemannian} Geometry},
  author={Ziheng Chen and Yue Song and Xiaojun Wu and Gaowen Liu and Nicu Sebe},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
}
```
This code is based on [DifferentiableSVD](https://github.com/KingJamesSong/DifferentiableSVD) from
- Why approximate matrix square root outperforms accurate SVD in global covariance pooling? [[Paper](https://arxiv.org/abs/2105.02498)]

We also acknowledge [fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV) from
- Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. [[Paper](https://arxiv.org/abs/1712.01034)]



If you have any problem, please contact me via `ziheng_ch@163.com`.

## Implementation
Please check `requirements.txt` for the required packages.

For dataset preparation, please refer to [fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV).

## Experiments
You can find all the scripts to compare Pow-EMLR, Pow-TMLR, and ScalePow-EMLR in `experiments.sh`. 
You can either execute all the experiments using `bash` or run each Python script individually.
We also incorporate Chol-TMLR in `src/representation/LCMCOV.py`.

