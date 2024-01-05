from torch.cuda import is_available as is_torch_cuda_available
from transformers.utils import is_torch_npu_available, is_torch_xpu_available

# https://github.com/huggingface/peft/blob/c4cf9e7d3b2948e71ec65a19e6cd1ff230781d13/src/peft/utils/other.py#L60-L71
def infer_device():
    if is_torch_cuda_available():
        torch_device = "cuda"
    elif is_torch_xpu_available():
        torch_device = "xpu"
    elif is_torch_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device
