# AEMatter [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-context-aggregation-in-natural/image-matting-on-composition-1k-1)](https://paperswithcode.com/sota/image-matting-on-composition-1k-1?p=rethinking-context-aggregation-in-natural)

Official repository for the paper [**Revisiting Context Aggregation for Image Matting**](https://arxiv.org/abs/2304.01171)

## Description

AEMatter is a simple yet powerful matting network. 
> 无有入无间，吾是以知无为之有益。
> 
> Only nothing can enter into no-space. Hence, I know the advantages of non-doing.

## Requirements
#### Hardware:

GPU memory >= 10GB for inference on Adobe Composition-1K testing set.

#### Packages:

- torch >= 1.10, < 2.0
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** It is recommended to use the RWA (Real World Augmentation) model for matting on real-world images.

Quantitative results on Adobe Composition-1K
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [AEMatter](https://mega.nz/file/7N4AEKrS#L4h3Cm2qLMMbwBGm1lyGOmVDTXJwDMAi4BlBauqNHrI) | 195MiB | 2.26 | 17.53 | 4.76 | 12.46 |
| [AEMatter+TTA](https://mega.nz/file/7N4AEKrS#L4h3Cm2qLMMbwBGm1lyGOmVDTXJwDMAi4BlBauqNHrI) | 195MiB | 2.06 | 16.89 | 4.24 | 11.72 |
| [AEMatter (RWA)](https://mega.nz/file/OEAhHAwB#jt_qn4v5RA1nNX4URDCjqDUA0Xu-UILRJJq9CCB13dk) | 195MiB | - | - | - | - |
| AEMatterV2 | - | 1.85 | 16.25 | 3.86 | 11.14 |

Quantitative results on Transparent-460
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| AEMatter | 195MiB |6.92|122.27|27.42|112.02 |

Quantitative results on AIM-500
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| AEMatter | 195MiB | 11.69 | 14.76 | 11.20 | 14.20 | 

Due to differences in data set preparation, the quantitative results on Distinction-646 and Semantic Image Matting are not shown.

## Training
We provide the script `train.py`  for training. You should modify the `dataset.py` file to set the data paths. The training and testing code appears to have numerical instability issues when executed on GPUs with PyTorch 2.0. This problem can be alleviated by modifying the order of the norm layers in AEAL. We have provided a [PyTorch 2.0 branch](https://github.com/aipixel/AEMatter/tree/Pytorch2.0), but it has not been trained or evaluated.

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


