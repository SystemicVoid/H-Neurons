import sys

import torch


assert torch.cuda.is_available(), "CUDA unavailable inside container"
device = torch.device("cuda:0")
x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)


@torch.compile(fullgraph=True, mode="reduce-overhead")
def matmul(t: torch.Tensor) -> torch.Tensor:
    return t @ t.T


y = matmul(x)
torch.cuda.synchronize()
print(
    f"[smoke] torch={torch.__version__} dev={torch.cuda.get_device_name(0)} "
    f"shape={tuple(y.shape)} ok=1"
)
sys.exit(0)
