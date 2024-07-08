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

- torch >= 2.0
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** It is recommended to use the RWA (Real World Augmentation) model for matting on real-world images.

## Training
We provide the script `train.py`  for training. You should modify the `dataset.py` file to set the data paths. 

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


