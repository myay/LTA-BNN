# TLU-BNN
A framework for evaluating the TLU-based execution of BNNs.

Supported:
- FashionMNIST CNN
- TLU computation for Linear and Conv2d layers
- Variable number of xnor gates
- Additional sampling windows (1-2 more)
- Variable majority vote shift 

TODOs:
- Error probability sampling from TLU-based execution
- Saving training state of BNN for retraining
- Other BNN models, e.g. CIFAR10 and CIFAR100

#### CUDA-based Binarization, TLU-based execution, and Error Injection

First, install PyTorch. For high performance, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

After successful installation of all kernels, run

```python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=10 --lr=0.001 --step-size=2 --gamma=0.5 --test-error```.

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
