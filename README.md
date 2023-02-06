# LTA-BNN
The framework for evaluating the LTA-based execution of BNNs, used in the paper with the title "Global by Local Thresholding in Binarized Neural Networksfor Efficient Crossbar Accelerator Design". Note that in the code, "LTA" and "TLU" are used synonymously.

Tested setups:
- Python 3.10.6 (pip + venv), PyTorch 1.13.1, GeForce RTX 3060 Ti 8GB (Driver Version: 525.60.13, CUDA Version: 12.0), Ubuntu 22.04.1
- Python 3.6.9, PyTorch 1.5.0, GeForce GTX 1060 6GB (Driver Version: 440.100, CUDA Version: 10.2), Ubuntu 18.04.4
- Python 3.6.13 (conda), 1.7.0+cu110, GeForce GTX 1080 8GB (Driver Version: 450.102.04, CUDA Version: 11.0), Ubuntu 18.04.6
- Python 3.9.7, PyTorch 1.9.0, GeForce GTX 3080 10GB (Driver Version: 512.15, CUDA Version: 11.6)

Supported:
- Datasets: FashionMNIST, KMNIST, SVHN, CIFAR10, IMAGENETTE
- BNN Models: VGG3, VGG7
- LTA computation for Linear and Conv2d layers
- Variable number of XNOR gates
- Additional sampling windows (1-2 more)
- Variable majority vote shift
- Threshold scaling (factor 2 supported)
- Threshold correction and custom threshold mechanisms
- (Re)Training with LTA-based execution

#### CUDA-based Binarization, LTA-based execution, and Error Injection

First, install PyTorch. For high performance, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

After successful installation of all kernels, for training run

```python3 run.py --model=VGG3 --dataset=FMNIST --train-model=1 --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gpu-num=0 --save-model="model_name"```.

Then, for LTA-based inference, run

```python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path="model_name.pt" --lta-mode=1 --nr-xnor-gates=64 --test-batch-size=1000 --gpu-num=0```.

For training based on LTA-execution, run

```python3 run.py --model=VGG3 --dataset=FMNIST --train-model=1 --lta-train=1 --lta-mode=1 --nr-xnor-gates=64 --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gpu-num=0 --save-model="FMNIST_TLU_TRAIN_64"```.

| Command line parameter | Options |
| :------------- |:-------------|
| --model      | FC, VGG3, VGG7 (for SVHN and CIFAR10), VGG7_L (for IMAGENETTE) |
| --dataset      | MNIST, FMNIST, KMNIST, SVHN, CIFAR10, IMAGENETTE |
| --lta-mode      | int, whether to use LTA-based inference, default: None |
| --nr-xnor-gates      | int, number of xnor gates in LTA execution, default: None |
| --error-prob      | float, probability for the outputs of the LTA approximation to flip, default: 0 |
| --train-model      | int, whether to train a model, default: None |
| --lossfunction      | MHL or CEL, default: MHL |
| --MHL-param      | int, parameter in MHL, default: 128|
| --lta-train      | int, whether LTA-based inference should be used in training, default: None |
| --epochs      | int, number of epochs to train, default: 10|
| --lr      | float, learning rate, default: 1.0|
| --gamma      | float, learning rate step, default: 0.5|
| --step-size      | int, learning rate step site, default: 5|
| --batch-size      | int, specifies the batch size in training, default: 64|
| --test-batch-size      | int, specifies the batch size in testing, default: 1000|
| --save-model | string, saves a trained model with the specified name in the string, default:None |
| --load-model-path | string, loads a model from the specified path in the string, default: None |
| --load-training-state | string, saves a training state with the specified name in the string, default:None |
| --save-training-state | string, loads a training state from the specified path in the string, default: None |
| --gpu-num | int, specifies the GPU on which the training should be performed, default: 0 |
| --profile-time | int, Specify whether to profile the execution time by specifying the repetitions, default: None |

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
