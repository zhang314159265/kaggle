# Reference
- [Competition Link](https://www.kaggle.com/competitions/cifar-10/overview)
- [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [An Example ResNet Model in PyTorch for cifar10](https://www.kaggle.com/code/toygarr/resnet-implementation-for-image-classification)
- [PyTorch official ResNet implementation](https://pytorch.org/hub/pytorch_vision_resnet/)
  - [code link](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

# Goal
0.5 score (median number)

# Results
- CNN
  - without normalization: 0.09980 (just like a random guess..)
  - with normalization: 0.41260 (not too bad)
- ResNet
  - BasicBlock on GPU 0.67720

For ResNet, GPU trains faster than CPU.
