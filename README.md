# Teacher-free-Knowledge-Distillation

## 1. Preparations

Clone this repository:

```
git clone https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation.git
```

### 1.1 Environment

Build a new environment and install:

```
pip install -r requirements.txt
```

### 1.2 Dataset

[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Tiny_ImageNet](https://tiny-imagenet.herokuapp.com/);
For CIFAR100 and CIFAR10, our codes will download the datasets automatically. For Tiny-ImageNet, you should download and put in the dir: "data/". The follow instruction and commands are for CIFAR100.

## 2. Train baseline models

You can skip this step by using our pre-trained models in [here](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation/releases/download/checkpoint/pretrained_teacher_models.zip). Download and unzip to: experiments/pretrained_teacher_models/

Use ''--model_dir'' to specify the directory of "parameters", model saving and log saving.

For example, normally train ResNet18 to obtain the pre-trained teacher:

```
CUDA_VISIBLE_DEVICES=0 python main.py --model_dir experiments/base_experiments/base_resnet18/
```

We ignore the command ''CUDA_VISIBLE_DEVICES=gpu_id'' in the following commands

Normally train MobileNetV2 to obtain the baseline model and baseline accuracy:

```
python main.py --model_dir experiments/base_experiments/base_mobilenetv2/
```

Normally train ResNeXt29 to obtain the baseline model and baseline accuracy:

```
python main.py --model_dir experiments/base_experiments/base_resnext29/
```

The baseline accuracy (in %) on CIFAR100 is:

| Model        | Baseline Acc |
| :----------- | :----------: |
| MobileNetV2  |    68.38     |
| ShuffleNetV2 |    70.34     |
| ResNet18     |    75.87     |
| ResNet50     |    78.16     |
| GoogLeNet    |    78.72     |
| Desenet121   |    79.04     |
| ResNeXt29    |    81.03     |

## 3. Exploratory experiments (Section 2 in our paper)

Normal KD: ResNet18 teach MobileNetV2

```
python main.py --model_dir experiments/kd_experiments/mobilenet_distill/resnet18_teacher/
```

### Reference

If you find this repo useful, please consider citing:

```
@inproceedings{yuan2020revisiting,
  title={Revisiting Knowledge Distillation via Label Smoothing Regularization},
  author={Yuan, Li and Tay, Francis EH and Li, Guilin and Wang, Tao and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3903--3911},
  year={2020}
}
```
