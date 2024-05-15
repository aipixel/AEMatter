# AEMatter [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-context-aggregation-in-natural/image-matting-on-composition-1k-1)](https://paperswithcode.com/sota/image-matting-on-composition-1k-1?p=rethinking-context-aggregation-in-natural)

Official repository for the paper [**Revisiting Context Aggregation for Image Matting**](https://arxiv.org/abs/2304.01171)

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

Quantitative results on Adobe Composition-1K
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [AEMatter](https://mega.nz/file/7N4AEKrS#L4h3Cm2qLMMbwBGm1lyGOmVDTXJwDMAi4BlBauqNHrI) | 195MiB | 2.26 | 17.53 | 4.76 | 12.46 |
| [AEMatter+TTA](https://mega.nz/file/7N4AEKrS#L4h3Cm2qLMMbwBGm1lyGOmVDTXJwDMAi4BlBauqNHrI) | 195MiB | 2.06 | 16.89 | 4.24 | 11.72 |
| [AEMatter (RWA)](https://mega.nz/file/OEAhHAwB#jt_qn4v5RA1nNX4URDCjqDUA0Xu-UILRJJq9CCB13dk) | 195MiB | - | - | - | - |

Quantitative results on Transparent-460
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| AEMatter | 195MiB |6.92|122.27|27.42|112.02 |

Quantitative results on AIM-500
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| AEMatter | 195MiB | 11.69 | 14.76 | 11.20 | 14.20 | 

Due to differences in data set preparation, the quantitative results on Distinction-646 and Semantic Image Matting are not shown.

## Evaluation
We provide the script `eval.py`  for evaluation.

## Citation
```
@inproceedings{liu2024aematter,
  title={Revisiting Context Aggregation for Image Matting},
  author={Liu, Qinglin and Lv, Xiaoqian and Meng, Quanling and Li, Zonglin and Lan, Xiangyuan and Yang, Shuo and Zhang, Shengping and Nie, Liqiang},
  booktitle ={International Conference on Machine Learning (ICML)},
  year={2024},
}
```


