# AEMatter 

Official repository for the paper [**Rethinking Context Aggregation in Natural Image Matting**](https://arxiv.org/abs/2304.01171)

## Description

AEMatter is a simple yet powerful matting network.

## Requirements
#### Hardware:

GPU memory >= 10GB for inference on Adobe Composition-1K testing set.

#### Packages:

- torch >= 1.10
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** 

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [AEMatter](https://pan.baidu.com/s/12p0YSnFsNpAZXHTGiLlarg?pwd=AEAL) | 208MiB | 2.39 | 17.79 | 4.81 | 12.64 |
| AEMatter+TTA | 208MiB | 2.20 | 17.43 | 4.31 | 12.21 |
## Evaluation
We provide the script `eval.py`  for evaluation.

## Additional experiments

We train IndexNet using the findings in this paper, and the [new implementation](https://github.com/QLYoo/YAIndexNet) achieves superior performance.

