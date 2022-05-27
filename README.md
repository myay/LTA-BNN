# LTA-BNN
A framework for evaluating the TLU-based execution of BNNs.

Tested setups:
- Python 3.6.9, PyTorch 1.5.0, GeForce GTX 1060 6GB (Driver Version: 440.100, CUDA Version: 10.2)
- Python 3.6.13 (conda), 1.7.0+cu110, GeForce GTX 1080 8GB (Driver Version: 450.102.04, CUDA Version: 11.0)
- Python 3.9.7, PyTorch 1.9.0, GeForce GTX 3080 10GB (Driver Version: 512.15, CUDA Version: 11.6)

Supported:
- FashionMNIST, KMNIST, SVHN, CIFAR10
- TLU computation for Linear and Conv2d layers
- Variable number of xnor gates
- Additional sampling windows (1-2 more)
- Variable majority vote shift
- Threshold scaling (factor 2 supported)
- Threshold correction and custom threshold mechanisms
- Training with TLU-based execution

TODOs:
- Saving training state of BNN for retraining
- Larger BNN models, e.g. CIFAR100

#### CUDA-based Binarization, TLU-based execution, and Error Injection

First, install PyTorch. For high performance, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

After successful installation of all kernels, for training run

```python3 run.py --model=VGG3 --dataset=FMNIST --train-model=1 --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gpu-num=0 --save-model="model_name"```.

Then, for TLU-based inference, run

```python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path="model_name.pt" --tlu-mode=1 --test-batch-size=1000 --gpu-num=0```.

For training based on TLU-execution, run

```python3 run.py --model=VGG3 --dataset=FMNIST --train-model=1 --tlu-train=1 --tlu-mode=1 --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gpu-num=0 --save-model="FMNIST_TLU_TRAIN"```.

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
