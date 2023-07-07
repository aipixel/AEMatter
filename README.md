# AEMatter 

Official repository for the paper [**Rethinking Context Aggregation in Natural Image Matting**](https://arxiv.org/abs/2304.01171)

## Description

AEMatter is a simple yet powerful matting network.
Since the implementation of Pytorch 2.x is different from Pytorch 1.x, resulting in unstable results, we optimize the network implementation and retrain the model using AMP. Additionally, we provide a model with RWA. The results are not fully tuned, but are still outstanding.

## Requirements
#### Hardware:

GPU memory >= 9GB for inference on Adobe Composition-1K testing set.

#### Packages:

- torch >= 2.0.0
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** 

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [AEMatter](https://mega.nz/file/nZRwQQhY#-XRJIPK4hKch2ZRvbc9owVpdBzoCeI818jK5MbjLC8o) | 208MiB | 2.35 | 18.40 | 4.69 | 13.43 |
| [AEMatter (RWA)](https://mega.nz/file/mVIUATIC#kBQhbHKq9op5KmCbQ5NB-klS7bpl8H_ba4PycsBlkiQ) | 208MiB | - | - | - | - |

## Evaluation
We provide the script `eval.py`  for evaluation. The model AEMatter work well on synthetic datasets, while the model AEMatter (RWA) work well on real-world images. It should be noted that these two models are trained to support Pytorch 2.0 (with AMP), but not fully tuned.

## Additional experiments

We train IndexNet using the findings in this paper, and the [new implementation](https://github.com/QLYoo/YAIndexNet) also achieves superior performance.

