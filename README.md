# Deep Residual Network
Tensorflow ResNet implementation on cifar10.  Please check out original github repo: [ResNet](https://github.com/KaimingHe/deep-residual-networks) and original paper: [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

## Dependencies
* Python (2.7.13)
* Tensorflow (1.2.1)
* matplotlib (2.0.2)
* numpy (1.13.1)

## Usage
**python Main.py** or use jupyter notebook to open **Main.ipynb** and run.

## Results
| Number of layers | Test error rate |
| :---------------:| :--------------:|
| 32               | 7.2%            |
| 110              | -               |

## Loss & error curve
32 layer
![missing error curve](https://github.com/jerryfan4/ResNet/blob/master/ResNet32/error.png)
![missing loss curve](https://github.com/jerryfan4/ResNet/blob/master/ResNet32/loss.png)