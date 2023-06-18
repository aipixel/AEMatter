# AEMatter 

Official repository for the paper [**Rethinking Context Aggregation in Natural Image Matting**](https://arxiv.org/abs/2304.01171)

## Description

AEMatter is a simple yet powerful matting network.
Since the implementation of Pytorch 2.x is different from Pytorch 1.x, resulting in unstable results, we optimize the network implementation and retrain the model using AMP. Additionally, we provide a model with RWA. The results are not fully tuned, but are still outstanding.

## Requirements
#### Hardware:

GPU memory >= 10GB for inference on Adobe Composition-1K testing set.

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
| [AEMatter]() | 208MiB | 2.39 | 17.79 | 4.81 | 12.64 |
| [AEMatter (RWA)]() | 208MiB | - | - | - | - |

## Evaluation
We provide the script `eval.py`  for evaluation.

## Additional experiments

We train IndexNet using the findings in this paper, and the [new implementation](https://github.com/QLYoo/YAIndexNet) also achieves superior performance.

