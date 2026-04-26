
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch F.linear - catches all nn.Linear calls
_orig_linear = F.linear
def _auto_linear(input, weight, bias=None):
    if isinstance(input, torch.Tensor) and input.is_floating_point() and input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    return _orig_linear(input, weight, bias)
F.linear = _auto_linear

# Patch torch.matmul - catches @ operator
_orig_matmul = torch.matmul
def _auto_matmul(input, other, *args, **kwargs):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        if input.is_floating_point() and other.is_floating_point() and input.dtype != other.dtype:
            input = input.to(other.dtype)
    return _orig_matmul(input, other, *args, **kwargs)
torch.matmul = _auto_matmul

# Patch torch.bmm
_orig_bmm = torch.bmm
def _auto_bmm(input, mat2, *args, **kwargs):
    if input.is_floating_point() and mat2.is_floating_point() and input.dtype != mat2.dtype:
        input = input.to(mat2.dtype)
    return _orig_bmm(input, mat2, *args, **kwargs)
torch.bmm = _auto_bmm

# Patch LayerNorm
_orig_ln = nn.LayerNorm.forward
def _auto_ln(self, input):
    return _orig_ln(self, input.to(self.weight.dtype))
nn.LayerNorm.forward = _auto_ln

# Patch GroupNorm
_orig_gn = nn.GroupNorm.forward
def _auto_gn(self, input):
    return _orig_gn(self, input.to(self.weight.dtype))
nn.GroupNorm.forward = _auto_gn

# Patch MultiheadAttention
_orig_mha = nn.MultiheadAttention.forward
def _auto_mha(self, query, key, value, **kwargs):
    try:
        dtype = next(self.parameters()).dtype
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    except StopIteration:
        pass
    return _orig_mha(self, query, key, value, **kwargs)
nn.MultiheadAttention.forward = _auto_mha

# Patch Conv layers
for _cls in [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d]:
    _orig = _cls.forward
    def _make_auto_conv(orig):
        def _auto_conv(self, input):
            return orig(self, input.to(self.weight.dtype))
        return _auto_conv
    _cls.forward = _make_auto_conv(_orig)

def register_auto_dtype_hooks(model):
    print("[SAM3 Native] Global dtype patches applied: F.linear, torch.matmul, torch.bmm, LayerNorm, GroupNorm, MHA, Conv")
    return []

# Patch @ operator on tensors
_orig_tensor_matmul = torch.Tensor.__matmul__
def _auto_tensor_matmul(self, other):
    if self.is_floating_point() and isinstance(other, torch.Tensor) and other.is_floating_point() and self.dtype != other.dtype:
        self = self.to(other.dtype)
    return _orig_tensor_matmul(self, other)
torch.Tensor.__matmul__ = _auto_tensor_matmul
