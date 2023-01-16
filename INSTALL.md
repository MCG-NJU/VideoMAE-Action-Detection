# Installation

The codebase is mainly built with following libraries:

- Python 3.8 or higher

- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>
  We can successfully reproduce the main results under this setting:<br>
  Tesla **A100** (40G): CUDA 11.1 + PyTorch 1.8.0 + torchvision 0.9.0<br>

- [timm==0.4.12](https://github.com/rwightman/pytorch-image-models)

- ~~[deepspeed==0.5.8](https://github.com/microsoft/DeepSpeed) `DS_BUILD_OPS=1 pip install deepspeed`~~

- [TensorboardX](https://github.com/lanpa/tensorboardX)

- [decord](https://github.com/dmlc/decord)

- [einops](https://github.com/arogozhnikov/einops)
  
- [PyAV](https://github.com/PyAV-Org/PyAV) `pip install av`
 

### Note:
- We recommend you to use **`PyTorch >= 1.8.0`**.
- We observed accidental interrupt in the last epoch when enabled `deepspeed`. Therefore, we disable the option of `deepspeed` in the released version.

