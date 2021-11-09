# learning-with-noisy-label

This is the initial code for the paper: A Model-Agnostic Approach for Learning with Noisy Labels of Arbitrary Distributions. After the paper is published, we will further organize the code and make it public.


## Setups

The requiring environment is as bellow:
- Linux
- Python 3+
- PyTorch 1.9.1
- Torchvision 0.10.1


## Runing on benchmark datasets (CIFAR-10 and CIFAR-100)

Here is an example of training ResNet50 model with our robust loss function Jc (10% class-dependent label noise will be injected into CIFAR-10):

```
python3 nar_main.py --data CIFAR10 --num_classes 10 --val_size 500 --model_name resnet --batch_size 16 --epochs 20 --error_rate 0.1 --error_type nar
```

Here is an example of training ResNet50 model with our robust loss function Jf (10% class-dependent label noise will be injected into CIFAR-10):

```
python3 nnar_main.py --data CIFAR10 --num_classes 10 --val_size 500 --model_name resnet --batch_size 16 --epochs 20 --error_rate 0.1 --error_type nnar --pretrained_model 'best_resnet_cifar10_nar10_unweight.pth.tar'
```
