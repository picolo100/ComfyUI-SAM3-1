# ComfyUI-SAM3.1

Native SAM3/SAM3.1 video masking nodes for ComfyUI — patched for NVIDIA Blackwell GPUs (RTX PRO 6000, RTX 5090, etc.) and fixed for stable multi-run inference without dtype crashes.

## What This Fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named 'flash_attn_interface'` | FA3 not available on Blackwell | Patched `is_hopper` detection |
| `offload_state_to_cpu unexpected kwarg` | Multiplex init_state signature mismatch | Patched `sam3_base_predictor.py` |
| `mat1 and mat2 must have same dtype` | Mixed bf16/fp32 tensors | Global `F.linear`, `torch.matmul`, `@` operator patches |
| `expected scalar type Float but found BFloat16` | LayerNorm/Conv dtype mismatch | Global nn.Module forward patches |
| `SAM 3.1 Object Multiplex builder not available` | Wrong sam3 loaded from comfyui-rmbg | Auto-renamed on install |
| `pkg_resources` not found | Python 3.12 removed it | `setuptools<70` enforced |

## Installation

Via ComfyUI Manager — search for `ComfyUI-SAM3.1`

Or manually:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/picolo100/ComfyUI-SAM3-1.git
cd ComfyUI-SAM3-1
pip install -r requirements.txt
```

Restart ComfyUI — the patched sam3 library installs automatically on first load.

## Model Download

```bash
wget "https://huggingface.co/research21/sam3.1/resolve/main/sam3.1_multiplex.pt" \
  -O ComfyUI/models/sam3/sam3.1_multiplex.pt
```

## Nodes

| Node | Description |
|---|---|
| SAM3 Native Model Loader (3.1 Multiplex) | Loads SAM3.1 model |
| SAM3 Native Video Mask (Text) | Segments objects via text prompt |
| SAM3 Native Video Mask (Points/Boxes) | Segments via point/box prompts |
| SAM3 Native Unload Model | Frees GPU memory |

## Recommended Settings

- `model`: `sam3.1_multiplex.pt`
- `predictor_type`: `auto`
- `precision`: `bf16`
- `warm_up`: `false`
- `memory_mode`: `auto`

## Credits

- Original SAM3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Original node: [Jalen-Brunson/ComfyUI-SAM3-Native](https://github.com/Jalen-Brunson/ComfyUI-SAM3-Native)

## License

Node code: MIT. SAM3 library: Meta SAM License (non-commercial research only).
