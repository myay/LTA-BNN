# TLU-BNN
A framework for evaluating the TLU-based execution of BNNs.

Supported:
- FashionMNIST CNN (VGG-3), CIFAR10 CNN (VGG-7)
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

```python3 run_fashion_binarized.py --model=VGG3 --dataset=FMNIST --train-model=1 --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gpu-num=0 --save-model="model_name.pt"```.

Then, for TLU-based inference, run

```python3 run_fashion_binarized.py --model=VGG3 --dataset=FMNIST --load-model-path="model_name.pt" --tlu-mode=1 --test-batch-size=1000 --gpu-num=0```.

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
