import torch
import torch.nn as nn
import torch.nn.functional as F

_orig_linear = F.linear
_orig_matmul = torch.matmul
_orig_bmm = torch.bmm
_orig_ln = nn.LayerNorm.forward
_orig_gn = nn.GroupNorm.forward
_orig_mha = nn.MultiheadAttention.forward
_orig_tensor_matmul = torch.Tensor.__matmul__
_orig_convs = {cls: cls.forward for cls in [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d]}

def _auto_linear(input, weight, bias=None):
    if isinstance(input, torch.Tensor) and input.is_floating_point() and input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    return _orig_linear(input, weight, bias)

def _auto_matmul(input, other, *args, **kwargs):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        if input.is_floating_point() and other.is_floating_point() and input.dtype != other.dtype:
            input = input.to(other.dtype)
    return _orig_matmul(input, other, *args, **kwargs)

def _auto_bmm(input, mat2, *args, **kwargs):
    if input.is_floating_point() and mat2.is_floating_point() and input.dtype != mat2.dtype:
        input = input.to(mat2.dtype)
    return _orig_bmm(input, mat2, *args, **kwargs)

def _auto_ln(self, input):
    if self.weight is not None:
        return _orig_ln(self, input.to(self.weight.dtype))
    return _orig_ln(self, input)

def _auto_gn(self, input):
    if self.weight is not None:
        return _orig_gn(self, input.to(self.weight.dtype))
    return _orig_gn(self, input)

def _auto_mha(self, query, key, value, **kwargs):
    try:
        dtype = next(self.parameters()).dtype
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    except StopIteration:
        pass
    return _orig_mha(self, query, key, value, **kwargs)

def _auto_tensor_matmul(self, other):
    if self.is_floating_point() and isinstance(other, torch.Tensor) and other.is_floating_point() and self.dtype != other.dtype:
        self = self.to(other.dtype)
    return _orig_tensor_matmul(self, other)

def enable_sam3_patches():
    F.linear = _auto_linear
    torch.matmul = _auto_matmul
    torch.bmm = _auto_bmm
    nn.LayerNorm.forward = _auto_ln
    nn.GroupNorm.forward = _auto_gn
    nn.MultiheadAttention.forward = _auto_mha
    torch.Tensor.__matmul__ = _auto_tensor_matmul
    for cls in [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d]:
        def _make(orig):
            def _auto_conv(self, input):
                if self.weight is not None:
                    return orig(self, input.to(self.weight.dtype))
                return orig(self, input)
            return _auto_conv
        cls.forward = _make(_orig_convs[cls])
    print("[SAM3 Native] dtype patches ENABLED")

def disable_sam3_patches():
    F.linear = _orig_linear
    torch.matmul = _orig_matmul
    torch.bmm = _orig_bmm
    nn.LayerNorm.forward = _orig_ln
    nn.GroupNorm.forward = _orig_gn
    nn.MultiheadAttention.forward = _orig_mha
    torch.Tensor.__matmul__ = _orig_tensor_matmul
    for cls, orig in _orig_convs.items():
        cls.forward = orig
    torch._dynamo.reset()
    print("[SAM3 Native] dtype patches DISABLED + dynamo reset")

def register_auto_dtype_hooks(model):
    print("[SAM3 Native] Conditional dtype patches ready (call enable_sam3_patches() before inference)")
    return []
