"""
ComfyUI-SAM3.1
=======================================
SAM3/SAM3.1 segmentation nodes using Facebook's native sam3 library.
Supports Object Multiplex for ~7x faster multi-object tracking.

Requires: pip install git+https://github.com/facebookresearch/sam3.git

IMPORTANT: For SAM 3.1 checkpoints (sam3.1_multiplex.pt), you MUST have
the latest code from the main branch (post March 27, 2026).

Model checkpoints should be placed in ComfyUI/models/sam3/:
- sam3.pt (SAM 3.0)
- sam3.1_multiplex.pt (SAM 3.1 with Object Multiplex)

Compatible with existing SAM3 nodes:
- Uses SAM3_MULTI_MASK for multi_mask output (works with Sam3ExtractObjectMask)
- Accepts SAM3_OBJECT_PROMPTS from Sam3ObjectPrompt nodes
"""

import os
import sys
import gc
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from collections import OrderedDict
from PIL import Image
import folder_paths
import comfy.model_management as mm

# Inference cache — avoids re-running propagation when downstream nodes re-execute
class _InferenceCache:
    def __init__(self, max_size=3):
        self._cache = OrderedDict()
        self._max_size = max_size

    def make_key(self, **kwargs) -> str:
        h = hashlib.sha256()
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                h.update(f"{k}:t:{v.shape}:{v.dtype}".encode())
                flat = v.flatten()
                if flat.numel() > 0:
                    h.update(f"{flat[0].item():.6f},{flat[flat.numel()//2].item():.6f},{flat[-1].item():.6f},{flat.sum().item():.4f}".encode())
            elif v is None:
                h.update(f"{k}:None".encode())
            else:
                h.update(f"{k}:{v}".encode())
        return h.hexdigest()[:16]

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

_inference_cache = _InferenceCache()

# =============================================================================
# CRITICAL: Ensure we use the official Facebook sam3 library
# =============================================================================
def _get_sam3_builders():
    """
    Import the official Facebook sam3 model builders.
    
    Returns a dict with:
    - "multiplex": build_sam3_multiplex_video_predictor (for SAM 3.1 Object Multiplex)
    - "standard": build_sam3_video_predictor (for SAM 3.0)
    
    SAM 3.1 Object Multiplex uses a DIFFERENT architecture and API than SAM 3.0!
    - sam3.1_multiplex.pt requires build_sam3_multiplex_video_predictor
    - sam3.pt requires build_sam3_video_predictor
    """
    # Check if sam3 is already imported from a bad location
    if 'sam3' in sys.modules:
        sam3_module = sys.modules['sam3']
        sam3_path = getattr(sam3_module, '__file__', '')
        
        # Detect if it's from a custom_nodes bundle (bad)
        if 'custom_nodes' in sam3_path:
            print(f"[SAM3 Native] WARNING: Found bundled sam3 at {sam3_path}")
            print(f"[SAM3 Native] This may be incompatible with SAM3.1 checkpoints.")
            print(f"[SAM3 Native] Consider removing it and installing the latest from GitHub.")
    
    builders = {}
    
    # Try to import
    try:
        from sam3 import model_builder
        
        sam3_path = sys.modules['sam3'].__file__
        print(f"[SAM3 Native] Using sam3 from: {sam3_path}")
        
        # Check for SAM 3.1 multiplex builder (required for sam3.1_multiplex.pt)
        if hasattr(model_builder, 'build_sam3_multiplex_video_predictor'):
            builders["multiplex"] = model_builder.build_sam3_multiplex_video_predictor
            print("[SAM3 Native] SAM 3.1 Object Multiplex support: AVAILABLE")
        else:
            print("[SAM3 Native] SAM 3.1 Object Multiplex support: NOT AVAILABLE")
            print("[SAM3 Native] You need the latest sam3 code from GitHub for SAM 3.1!")
            print("[SAM3 Native] Run: cd /tmp && git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .")
        
        # Standard SAM 3.0 builder
        if hasattr(model_builder, 'build_sam3_video_predictor'):
            builders["standard"] = model_builder.build_sam3_video_predictor
            
        if not builders:
            raise ImportError("No sam3 video predictor builders found!")
            
        return builders
        
    except ImportError as e:
        raise ImportError(
            f"Could not import sam3 library: {e}\n"
            f"Please install the official Facebook sam3:\n"
            f"  cd /tmp && git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .\n"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading sam3: {e}")


# Constants
SAM3_MODEL_DIR = "sam3"
DEFAULT_MODEL = "sam3.1_multiplex.pt"


def normalize_object_prompts_for_native(object_prompts, frame_w: int, frame_h: int) -> List[Dict]:
    """
    Convert SAM3_OBJECT_PROMPTS format to native sam3 API format.
    
    SAM3_OBJECT_PROMPTS format (from Sam3ObjectPrompt):
    [
        {
            "positive_points": {"points": [[x, y], ...]},
            "negative_points": {"points": [[x, y], ...]},
            "positive_boxes": {"boxes": [[x1, y1, x2, y2], ...]},
            "_pixel_coords": True/False
        },
        ...
    ]
    
    Native SAM 3.1 API expects RELATIVE coordinates (0-1 range).
    Returns list of dicts ready for add_prompt calls.
    """
    if object_prompts is None:
        return []
    
    result = []
    for i, prompt in enumerate(object_prompts):
        native_prompt = {
            "object_id": i + 1,  # Assign object IDs starting from 1
            "points": [],       # [[x, y], ...] in relative coords
            "labels": [],       # [1, 1, 0, ...], 1=positive, 0=negative
            "box": None,        # [x, y, w, h] in relative coords (optional)
        }
        
        is_pixel = prompt.get("_pixel_coords", False)
        
        # Process positive points -> convert to relative coords
        pos_points = prompt.get("positive_points", {}).get("points", [])
        for pt in pos_points:
            x, y = pt[0], pt[1]
            if is_pixel:
                # Convert pixel to relative
                x = x / frame_w
                y = y / frame_h
            native_prompt["points"].append([x, y])
            native_prompt["labels"].append(1)
        
        # Process negative points -> convert to relative coords
        neg_points = prompt.get("negative_points", {}).get("points", [])
        for pt in neg_points:
            x, y = pt[0], pt[1]
            if is_pixel:
                x = x / frame_w
                y = y / frame_h
            native_prompt["points"].append([x, y])
            native_prompt["labels"].append(0)
        
        # Process boxes (use first box) -> convert to relative coords
        boxes = prompt.get("positive_boxes", {}).get("boxes", [])
        if boxes:
            box = boxes[0]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            if is_pixel:
                x1, x2 = x1 / frame_w, x2 / frame_w
                y1, y2 = y1 / frame_h, y2 / frame_h
            # SAM 3.1 uses [x, y, w, h] format for boxes
            w = x2 - x1
            h = y2 - y1
            native_prompt["box"] = [x1, y1, w, h]
        
        if native_prompt["points"] or native_prompt["box"]:
            result.append(native_prompt)
    
    return result


def normalize_multi_prompts_for_native(multi_prompts, frame_w: int, frame_h: int) -> List[Dict]:
    """
    Convert SAM3_MULTI_PROMPTS format (from SAM3MultiRegionCollector) to native sam3 API format.
    
    SAM3_MULTI_PROMPTS format (from SAM3MultiRegionCollector):
    [
        {
            "id": 0,
            "positive_points": {"points": [[x, y], ...], "labels": [1, 1, ...]},
            "negative_points": {"points": [[x, y], ...], "labels": [0, 0, ...]},
            "positive_boxes": {"boxes": [...], "labels": [...]},
            "negative_boxes": {"boxes": [...], "labels": [...]}
        },
        ...
    ]
    
    NOTE: The points are already in RELATIVE coordinates (0-1 range).
    
    Returns list of dicts ready for add_prompt calls.
    """
    if multi_prompts is None:
        return []
    
    # Handle string input (JSON encoded from widget)
    if isinstance(multi_prompts, str):
        try:
            import json
            multi_prompts = json.loads(multi_prompts)
        except:
            print(f"[SAM3 Native] Failed to parse multi_prompts as JSON: {multi_prompts[:100] if len(multi_prompts) > 100 else multi_prompts}")
            return []
    
    result = []
    for i, prompt in enumerate(multi_prompts):
        # Use the 'id' field if present, otherwise use index
        obj_id = prompt.get("id", i) + 1  # Object IDs start from 1
        
        native_prompt = {
            "object_id": obj_id,
            "points": [],       # [[x, y], ...] in relative coords
            "labels": [],       # [1, 1, 0, ...], 1=positive, 0=negative
            "box": None,        # [x, y, w, h] in relative coords (optional)
        }
        
        # Process positive points - already in relative coords
        pos_data = prompt.get("positive_points", {})
        pos_points = pos_data.get("points", []) if isinstance(pos_data, dict) else []
        for pt in pos_points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                native_prompt["points"].append([x, y])
                native_prompt["labels"].append(1)
        
        # Process negative points - already in relative coords
        neg_data = prompt.get("negative_points", {})
        neg_points = neg_data.get("points", []) if isinstance(neg_data, dict) else []
        for pt in neg_points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                native_prompt["points"].append([x, y])
                native_prompt["labels"].append(0)
        
        # Process boxes (use first positive box) - already in relative coords
        # SAM3MultiRegionCollector outputs boxes as [cx, cy, w, h] (center format)
        box_data = prompt.get("positive_boxes", {})
        pos_boxes = box_data.get("boxes", []) if isinstance(box_data, dict) else []
        if pos_boxes and len(pos_boxes) > 0:
            box = pos_boxes[0]
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                # Convert center format [cx, cy, w, h] to corner format [x, y, w, h] for SAM 3.1
                x = cx - bw / 2
                y = cy - bh / 2
                native_prompt["box"] = [x, y, bw, bh]
        
        if native_prompt["points"] or native_prompt["box"]:
            result.append(native_prompt)
            print(f"[SAM3 Native] Multi-prompt {obj_id}: {len(native_prompt['points'])} points, box={native_prompt['box'] is not None}")
    
    return result


def get_sam3_model_path():
    """Get the path to SAM3 models directory."""
    return os.path.join(folder_paths.models_dir, SAM3_MODEL_DIR)


def get_bpe_vocab_path() -> str:
    """
    Get path to BPE vocabulary file, downloading if necessary.
    The SAM3 native library requires this file for text encoding.
    """
    # First check if it exists in the sam3 package
    try:
        import sam3
        sam3_root = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(bpe_path):
            return bpe_path
    except:
        pass
    
    # Check in models/sam3 directory
    model_dir = get_sam3_model_path()
    local_bpe = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(local_bpe):
        return local_bpe
    
    # Download from HuggingFace
    print("[SAM3 Native] BPE vocab file not found, downloading...")
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_dir, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id="LanguageBind/LanguageBind",
            filename="open_clip/bpe_simple_vocab_16e6.txt.gz",
            repo_type="space",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.exists(downloaded_path) and downloaded_path != local_bpe:
            import shutil
            shutil.copy(downloaded_path, local_bpe)
        print(f"[SAM3 Native] BPE vocab downloaded to: {local_bpe}")
        return local_bpe
    except Exception as e:
        print(f"[SAM3 Native] Failed to download BPE vocab: {e}")
        return None


def list_sam3_models():
    """List available SAM3 model files."""
    model_dir = get_sam3_model_path()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return [DEFAULT_MODEL]
    
    models = []
    for f in os.listdir(model_dir):
        if f.endswith('.pt') and 'sam3' in f.lower():
            models.append(f)
    
    if not models:
        models = [DEFAULT_MODEL, "sam3.pt"]
    
    # Sort with 3.1 first
    models.sort(key=lambda x: (0 if '3.1' in x else 1, x))
    return models


def download_sam3_model(model_name: str) -> str:
    """Download SAM3 model from HuggingFace if not present."""
    model_dir = get_sam3_model_path()
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine HuggingFace repo based on model name
    if "3.1" in model_name or "multiplex" in model_name.lower():
        repo_id = "facebook/sam3.1"
        hf_filename = "sam3.1_multiplex.pt" if "multiplex" in model_name else model_name
    else:
        repo_id = "facebook/sam3"
        hf_filename = model_name
    
    print(f"[SAM3 Native] Model not found, downloading from HuggingFace...")
    print(f"[SAM3 Native] Repo: {repo_id}, File: {hf_filename}")
    
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[SAM3 Native] Model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[SAM3 Native] Download failed: {e}")
        print(f"[SAM3 Native] Please manually download {model_name} from https://huggingface.co/{repo_id}")
        raise


class LoadNativeSAM3VideoModel:
    """
    Load SAM3/SAM3.1 Video model using Facebook's native sam3 library.
    
    Supports:
    - SAM 3.0: Standard video tracking
    - SAM 3.1: Object Multiplex for ~7x faster multi-object tracking
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list_sam3_models(), {"default": DEFAULT_MODEL}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "gpus_to_use": ("STRING", {"default": "0", "multiline": False}),
                "predictor_type": (["auto", "multiplex", "standard"], {"default": "auto"}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "warm_up": ("BOOLEAN", {"default": False}),
                "memory_mode": (["auto", "speed", "balanced", "low_memory"], {"default": "auto"}),
                "num_maskmem": ("INT", {"default": 7, "min": 1, "max": 16, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SAM3NATIVEMODEL",)
    RETURN_NAMES = ("sam3_native_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3.1 Native/loaders"
    
    DESCRIPTION = """
    Load SAM3/SAM3.1 using Facebook's native library.

    Models:
    - sam3.1_multiplex.pt: SAM 3.1 with Object Multiplex (~7x faster)
    - sam3.pt: Original SAM 3.0

    predictor_type:
    - auto: multiplex for 3.1 checkpoints, standard for 3.0
    - multiplex: build_sam3_multiplex_video_predictor (~7x faster)
    - standard: build_sam3_video_predictor

    torch_compile: Enable torch.compile for ~10-20% speedup (first run slower)
    warm_up: Pre-compile with dummy inference to avoid first-run slowness (requires torch_compile)

    memory_mode:
    - auto: speed on H200 (>=90GB VRAM), balanced on H100 (>=70GB), low_memory otherwise
    - speed: Keep states on GPU, no memory trimming. Fastest but uses most VRAM.
    - balanced: Offload outputs to CPU, trim old memory. Good for H100/long videos.
    - low_memory: Offload everything to CPU. Slowest but handles very long videos.

    num_maskmem: Number of past frames kept in memory bank (default 7).
    Lower values = faster inference + less VRAM. The tracker attends to this many
    recent frames when predicting each new frame. 3-4 works well for smooth video,
    7 is conservative. Does NOT skip frames — every frame is still processed.

    gpus_to_use: Comma-separated GPU IDs (e.g., "0" or "0,1")
    """
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities for optimization."""
        if not torch.cuda.is_available():
            return {"has_cuda": False}
        
        props = torch.cuda.get_device_properties(0)
        compute_cap = (props.major, props.minor)
        
        info = {
            "has_cuda": True,
            "name": props.name,
            "compute_capability": compute_cap,
            "is_hopper": compute_cap[0] == 9,  # Hopper only, not Blackwell (12)
            "is_ampere": compute_cap[0] >= 8,
            "total_memory_gb": props.total_memory / (1024**3),
            "has_bf16": torch.cuda.is_bf16_supported(),
        }
        
        return info
    
    def load_model(self, model: str, device: str, precision: str,
                   gpus_to_use: str = "0", predictor_type: str = "auto",
                   torch_compile: bool = False, warm_up: bool = False,
                   memory_mode: str = "auto", num_maskmem: int = 7):
        
        gpu_info = self._detect_gpu_capabilities()
        
        if device == "cuda" and not gpu_info["has_cuda"]:
            print("[SAM3 Native] CUDA not available, falling back to CPU")
            device = "cpu"
        
        if device == "cuda":
            print(f"[SAM3 Native] GPU: {gpu_info['name']}")
            print(f"[SAM3 Native] Compute Capability: {gpu_info['compute_capability']}")
            print(f"[SAM3 Native] VRAM: {gpu_info['total_memory_gb']:.1f} GB")
            if gpu_info["is_hopper"]:
                print("[SAM3 Native] Hopper architecture detected - FA3 available")

            # Disable CUDA graph trees in torch.inductor to avoid
            # "cudaMallocAsync does not yet support checkPoolLiveAllocations" error.
            # SAM3 lib internally uses torch.compile with max-autotune for nms_masks,
            # which triggers CUDA graphs that conflict with cudaMallocAsync.
            try:
                import torch._inductor.config as inductor_config
                inductor_config.triton.cudagraph_trees = False
                print("[SAM3 Native] Disabled inductor CUDA graph trees (cudaMallocAsync compat)")
            except Exception:
                pass
        
        # Parse GPU IDs
        gpu_ids = [int(g.strip()) for g in gpus_to_use.split(",") if g.strip()]
        if not gpu_ids:
            gpu_ids = [0]
        
        # Get model path
        model_dir = get_sam3_model_path()
        model_path = os.path.join(model_dir, model)
        
        if not os.path.exists(model_path):
            model_path = download_sam3_model(model)
        
        print(f"[SAM3 Native] Loading model: {model}")
        print(f"[SAM3 Native] Precision: {precision}")
        print(f"[SAM3 Native] GPUs: {gpu_ids}")
        
        # Enable optimizations
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Get BPE vocabulary path (required for text encoding)
        bpe_path = get_bpe_vocab_path()
        if bpe_path:
            print(f"[SAM3 Native] Using BPE vocab: {bpe_path}")
        
        # Import sam3 builders
        builders = _get_sam3_builders()
        
        # Determine which builder to use
        is_multiplex_checkpoint = "multiplex" in model.lower() or "3.1" in model
        
        # Resolve predictor type
        if predictor_type == "auto":
            use_multiplex = is_multiplex_checkpoint
        elif predictor_type == "multiplex":
            use_multiplex = True
        else:  # "standard"
            use_multiplex = False
        
        if use_multiplex:
            if "multiplex" not in builders:
                raise RuntimeError(
                    f"SAM 3.1 Object Multiplex builder not available!\n\n"
                    f"Your sam3 library doesn't have build_sam3_multiplex_video_predictor.\n\n"
                    f"To fix this, install the latest sam3 from GitHub main branch:\n"
                    f"  pip uninstall sam3 -y\n"
                    f"  cd /tmp && rm -rf sam3\n"
                    f"  git clone https://github.com/facebookresearch/sam3.git\n"
                    f"  cd sam3 && pip install -e .\n\n"
                    f"Then restart ComfyUI."
                )
            build_fn = builders["multiplex"]
            print(f"[SAM3 Native] Using SAM 3.1 MULTIPLEX builder (build_sam3_multiplex_video_predictor)")
        else:
            if "standard" not in builders:
                raise RuntimeError("SAM 3.0 standard builder not available in sam3 library!")
            build_fn = builders["standard"]
            print(f"[SAM3 Native] Using STANDARD builder (build_sam3_video_predictor)")
        
        # Build the video predictor
        # SAM 3.1 build_sam3_multiplex_video_predictor() takes NO ARGUMENTS
        # It auto-downloads from HuggingFace. For custom checkpoints, we need to
        # either use symlinks or check if a checkpoint_path arg is supported.
        
        predictor = None
        
        # First, check the function signature
        import inspect
        try:
            sig = inspect.signature(build_fn)
            params = list(sig.parameters.keys())
            print(f"[SAM3 Native] Builder function parameters: {params}")
        except:
            params = []
        
        # Build optional kwargs based on what the builder accepts
        optional_kwargs = {}
        if "use_fa3" in params:
            use_fa3 = gpu_info.get("is_hopper", False)
            optional_kwargs["use_fa3"] = use_fa3
            print(f"[SAM3 Native] Flash Attention 3: {use_fa3}")
        # NEVER pass compile/warm_up to the builder — they can cause
        # unrecoverable SIGABRT crashes that kill the entire ComfyUI process.
        # torch.compile is applied separately after a successful load.
        if torch_compile:
            print(f"[SAM3 Native] torch.compile will be applied AFTER model load (safe mode)")

        # Try different calling patterns
        if "checkpoint_path" in params or "checkpoint" in params:
            ckpt_arg = "checkpoint_path" if "checkpoint_path" in params else "checkpoint"
            try:
                kwargs = {ckpt_arg: model_path}
                if "gpus_to_use" in params:
                    kwargs["gpus_to_use"] = gpu_ids
                if "bpe_path" in params and bpe_path:
                    kwargs["bpe_path"] = bpe_path
                kwargs.update(optional_kwargs)
                print(f"[SAM3 Native] Loading model (suppressing verbose state_dict output)...")
                import io, contextlib
                _buf = io.StringIO()
                with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
                    predictor = build_fn(**kwargs)
                for _line in _buf.getvalue().splitlines():
                    if _line.strip() and '.weight' not in _line and '.bias' not in _line:
                        print(f"[SAM3 Native] {_line.strip()}")
            except Exception as e:
                print(f"[SAM3 Native] Failed with kwargs: {e}")
                predictor = None

        if predictor is None:
            try:
                # Fallback: never include compile/warm_up (they can SIGABRT)
                if optional_kwargs:
                    print(f"[SAM3 Native] Loading model with: {list(optional_kwargs.keys())} (suppressing verbose output)...")
                else:
                    print(f"[SAM3 Native] Loading model (suppressing verbose output)...")
                _buf2 = io.StringIO()
                with contextlib.redirect_stdout(_buf2), contextlib.redirect_stderr(_buf2):
                    predictor = build_fn(**optional_kwargs) if optional_kwargs else build_fn()
                for _line in _buf2.getvalue().splitlines():
                    if _line.strip() and '.weight' not in _line and '.bias' not in _line:
                        print(f"[SAM3 Native] {_line.strip()}")
            except Exception as e:
                error_msg = str(e)
                if "Missing key(s) in state_dict" in error_msg or "Unexpected key(s)" in error_msg:
                    raise RuntimeError(
                        f"Model architecture mismatch!\n\n"
                        f"Checkpoint: {model}\n"
                        f"The checkpoint doesn't match the library's expected architecture.\n\n"
                        f"This usually means:\n"
                        f"1. Your sam3 library is outdated (pre-March 27, 2026 SAM 3.1 release)\n"
                        f"2. The checkpoint file is corrupted or from a different source\n\n"
                        f"To fix:\n"
                        f"1. Install the latest sam3 from GitHub main branch:\n"
                        f"   pip uninstall sam3 -y\n"
                        f"   cd /tmp && rm -rf sam3\n"
                        f"   git clone https://github.com/facebookresearch/sam3.git\n"
                        f"   cd sam3 && pip install -e .\n\n"
                        f"2. Re-download the checkpoint from facebook/sam3.1 on HuggingFace\n\n"
                        f"3. Restart ComfyUI\n\n"
                        f"Original error (truncated): {error_msg[:500]}..."
                    )
                raise
        
        # The SAM3.1 predictor enables bf16 autocast globally
        # (torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        # so float32 image tensors are automatically cast to bf16 during inference.
        # Do NOT convert model to float32 — it kills FA3 performance.
        try:
            if hasattr(predictor, 'model'):
                model_obj = predictor.model
                model_dtype = None
                for param in model_obj.parameters():
                    model_dtype = param.dtype
                    break
                print(f"[SAM3 Native] Model dtype: {model_dtype} (bf16 autocast handles conversion)")
        except Exception as e:
            print(f"[SAM3 Native] Warning: Could not check model dtype: {e}")
        
        # Apply torch.compile AFTER successful load to avoid SIGABRT
        if torch_compile and hasattr(predictor, 'model'):
            try:
                print(f"[SAM3 Native] Applying torch.compile to loaded model...")
                predictor.model = torch.compile(predictor.model)
                print(f"[SAM3 Native] torch.compile applied successfully")
            except Exception as e:
                print(f"[SAM3 Native] torch.compile failed (non-fatal): {e}")

        # Map precision to dtype for autocast during inference
        precision_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        autocast_dtype = precision_map.get(precision, torch.bfloat16)

        # Resolve memory_mode "auto" based on GPU VRAM
        if memory_mode == "auto":
            vram_gb = gpu_info.get("total_memory_gb", 0)
            if vram_gb >= 90:
                memory_mode = "speed"
            elif vram_gb >= 70:
                memory_mode = "balanced"
            else:
                memory_mode = "low_memory"
            print(f"[SAM3 Native] memory_mode auto-resolved to '{memory_mode}' ({vram_gb:.0f}GB VRAM)")
        else:
            print(f"[SAM3 Native] memory_mode: {memory_mode}")

        offload_to_cpu = memory_mode in ("balanced", "low_memory")

        # Monkey-patch: disable non-overlapping mask constraints so that
        # overlapping objects (e.g. head + hair) each keep their full mask.
        # The native SAM3.1 library forces mutually-exclusive masks by default,
        # but we need independent per-object masks like the transformers API.
        try:
            tracker = predictor.model.tracker
            # Allow overlapping masks across objects (head + hair, etc.)
            tracker._apply_object_wise_non_overlapping_constraints = (
                lambda pred_masks, *args, **kwargs: pred_masks
            )
            # num_maskmem: how many past frames the tracker attends to per frame
            if tracker.num_maskmem != num_maskmem:
                print(f"[SAM3 Native] num_maskmem: {tracker.num_maskmem} → {num_maskmem}")
                tracker.num_maskmem = num_maskmem
            else:
                print(f"[SAM3 Native] num_maskmem: {num_maskmem}")
            # Memory trimming: always on (low overhead, prevents unbounded growth)
            tracker.trim_past_non_cond_mem_for_eval = True
            # CPU offload of tracker outputs: only when not in speed mode
            tracker.offload_output_to_cpu_for_eval = offload_to_cpu
            print(f"[SAM3 Native] Applied tracker patches: no-overlap-disabled, trim-mem, offload-output={offload_to_cpu}")

            if offload_to_cpu:
                # Patch _init_new_sam2_state to offload SAM2 tracker state to CPU
                # (prevents VRAM accumulation over thousands of frames)
                _orig_init_sam2 = predictor.model._init_new_sam2_state
                def _patched_init_sam2(inference_state):
                    state = _orig_init_sam2(inference_state)
                    state["offload_state_to_cpu"] = True
                    state["storage_device"] = torch.device("cpu")
                    return state
                predictor.model._init_new_sam2_state = _patched_init_sam2
                print(f"[SAM3 Native] Patched SAM2 state to offload to CPU")

                # Patch multiplex model's init_state so that per-object singleton
                # states also offload to CPU. Without this, the multiplex tracker
                # calls init_state() with offload_state_to_cpu=False (default)
                # for each object, causing VRAM accumulation across 1000s of frames.
                _multiplex_model = predictor.model
                _orig_init_state = _multiplex_model.init_state
                import inspect as _inspect
                _init_params = set(_inspect.signature(_orig_init_state).parameters.keys())
                def _patched_init_state(*args, **kwargs):
                    # Only inject offload_state_to_cpu if the underlying init_state
                    # accepts it — Sam3MultiplexTrackingWithInteractivity doesn't,
                    # but Sam3VideoTrackingMultiplexDemo does.
                    if "offload_state_to_cpu" in _init_params:
                        kwargs.setdefault("offload_state_to_cpu", True)
                    state = _orig_init_state(*args, **kwargs)
                    # Regardless of signature, force the state dict values
                    state["offload_state_to_cpu"] = True
                    state["storage_device"] = torch.device("cpu")
                    return state
                _multiplex_model.init_state = _patched_init_state
                print(f"[SAM3 Native] Patched multiplex init_state to offload singleton states to CPU")
            else:
                print(f"[SAM3 Native] Speed mode: keeping SAM2 state on GPU")
        except AttributeError as e:
            print(f"[SAM3 Native] Warning: Could not apply tracker patches: {e}")

        predictor.model.bfloat16()
        for buf_name, buf in predictor.model.named_buffers():
            if buf.is_floating_point():
                buf.data = buf.data.bfloat16()
        print("[SAM3 Native] Model + buffers converted to bf16")
        # Force all ops to bf16 globally
        # removed set_default_dtype
        print("[SAM3 Native] Set global default dtype to bf16")
        from sam3.auto_dtype import register_auto_dtype_hooks
        register_auto_dtype_hooks(predictor.model)
        print("[SAM3 Native] Model weights set to bf16")
        print("[SAM3 Native] Model weights set to bf16")
        print("[SAM3 Native] Model converted to bf16")
        print(f"[SAM3 Native] Model loaded successfully")
        print(f"[SAM3 Native] Predictor type: {'multiplex' if use_multiplex else 'standard'}")

        return ({
            "predictor": predictor,
            "model_name": model,
            "device": device,
            "gpu_ids": gpu_ids,
            "is_sam31": is_multiplex_checkpoint,
            "is_multiplex": use_multiplex,
            "gpu_info": gpu_info,
            "autocast_dtype": autocast_dtype,
            "memory_mode": memory_mode,
            "offload_to_cpu": offload_to_cpu,
        },)


class NativeSam3VideoMask:
    """
    Segment and track objects in video using text prompts.
    Uses Facebook's native SAM3 library for maximum performance.
    
    Outputs SAM3_MULTI_MASK compatible with Sam3ExtractObjectMask.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_native_model": ("SAM3NATIVEMODEL",),
                "video_frames": ("IMAGE",),
                "text_prompt": ("STRING", {"default": "person", "multiline": False}),
            },
            "optional": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MASK", "BBOX", "STRING", "SAM3_MULTI_MASK")
    RETURN_NAMES = ("masks", "bboxes", "detected_object_ids", "multi_mask")
    FUNCTION = "process_video"
    CATEGORY = "SAM3.1 Native/video"
    
    DESCRIPTION = """
    Segment objects in video using text prompt with native SAM3 library.
    
    SAM 3.1 with Object Multiplex provides ~7x faster multi-object tracking.
    
    text_prompt: What to segment (e.g., "person", "dog", "car")
    frame_index: Frame to add initial prompt
    threshold: Mask confidence threshold
    
    Outputs:
    - masks: Combined mask of all objects
    - bboxes: Bounding boxes
    - detected_object_ids: Comma-separated IDs
    - multi_mask: Per-object masks for Sam3ExtractObjectMask
    """
    
    def process_video(self, sam3_native_model: Dict, video_frames: torch.Tensor,
                      text_prompt: str, frame_index: int = 0,
                      threshold: float = 0.5, keep_model_loaded: bool = True):
        
        predictor = sam3_native_model["predictor"]
        is_sam31 = sam3_native_model["is_sam31"]
        
        print(f"[SAM3 Native] Processing video with text prompt: '{text_prompt}'")
        print(f"[SAM3 Native] SAM 3.1 Multiplex: {is_sam31}")
        
        # Get video dimensions
        num_frames = video_frames.shape[0]
        frame_h, frame_w = video_frames.shape[1], video_frames.shape[2]
        
        print(f"[SAM3 Native] Video: {num_frames} frames, {frame_w}x{frame_h}")
        
        offload_to_cpu = sam3_native_model.get("offload_to_cpu", True)
        memory_mode = sam3_native_model.get("memory_mode", "balanced")

        try:
            # Save frames to disk for SAM3's async frame loader (lazy per-frame loading).
            # Use torch uint8 conversion to avoid numpy float64 intermediate.
            import tempfile, shutil
            import time as _time
            _t0 = _time.perf_counter()
            temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
            print(f"[SAM3 Native] Saving {num_frames} frames to {temp_dir}...")
            video_uint8 = video_frames.cpu().mul(255).clamp_(0, 255).byte()

            from concurrent.futures import ThreadPoolExecutor
            def _save_frame(args):
                idx, frame_np = args
                Image.fromarray(frame_np).save(
                    os.path.join(temp_dir, f"{idx:05d}.jpg"), quality=80
                )
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
                pool.map(_save_frame, ((i, video_uint8[i].numpy()) for i in range(num_frames)))
            _t1 = _time.perf_counter()
            print(f"[SAM3 Native] Saved {num_frames} frames in {_t1 - _t0:.1f}s")
            del video_uint8
            gc.collect()

            # Start session with folder path — SAM3 uses AsyncImageFrameLoader
            print(f"[SAM3 Native] Starting tracking session...")
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_dir,
                    offload_video_to_cpu=True,
                )
            )
            session_id = response["session_id"]

            # Conditionally offload inference state based on memory_mode
            _session_data = predictor._all_inference_states.get(session_id)
            _inf_state = _session_data["state"] if _session_data else None
            if _inf_state is not None and offload_to_cpu:
                _inf_state["offload_state_to_cpu"] = True
                _inf_state["storage_device"] = torch.device("cpu")
                print(f"[SAM3 Native] Patched inference state: storage_device=cpu")
            elif _inf_state is not None:
                print(f"[SAM3 Native] Speed mode: inference state stays on GPU")

            # Add text prompt using SAM 3.1 API
            print(f"[SAM3 Native] Adding text prompt on frame {frame_index}...")
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=text_prompt,
                )
            )
            
            # Get initial detection results
            outputs = response.get("outputs", {})
            detected_obj_ids = set()
            
            if "out_obj_ids" in outputs:
                for oid in outputs["out_obj_ids"]:
                    detected_obj_ids.add(int(oid))
            
            print(f"[SAM3 Native] Detected {len(detected_obj_ids)} objects: {sorted(detected_obj_ids)}")
            
            # Propagate through video using STREAMING API
            # This is critical - SAM 3.1 uses handle_stream_request for propagation
            print(f"[SAM3 Native] Propagating masks through video...")
            outputs_per_frame = {}
            
            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]
                outputs_per_frame[frame_idx] = response["outputs"]
            
            print(f"[SAM3 Native] Propagation complete, got {len(outputs_per_frame)} frames")
            
            # Build output masks and multi_mask structure
            print(f"[SAM3 Native] Building output masks...")
            output_masks = []
            all_boxes = []
            masks_per_frame_per_obj = {}  # For SAM3_MULTI_MASK compatibility
            
            for frame_idx in range(num_frames):
                masks_per_frame_per_obj[frame_idx] = {}
                
                if frame_idx in outputs_per_frame:
                    frame_out = outputs_per_frame[frame_idx]
                    
                    # Debug: print frame output keys for first frame
                    if frame_idx == 0:
                        print(f"[SAM3 Native] Frame 0 output keys: {list(frame_out.keys())}")
                        for k, v in frame_out.items():
                            if hasattr(v, 'shape'):
                                print(f"[SAM3 Native]   {k}: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
                            elif hasattr(v, '__len__') and not isinstance(v, str):
                                print(f"[SAM3 Native]   {k}: len={len(v)}, type={type(v).__name__}")
                            else:
                                print(f"[SAM3 Native]   {k}: {v}")
                    
                    # Try different possible mask keys - SAM 3.1 uses out_binary_masks
                    masks = frame_out.get("out_mask_logits", None)
                    use_binary = False
                    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
                        masks = frame_out.get("out_binary_masks", None)
                        use_binary = True
                    
                    obj_ids = frame_out.get("out_obj_ids", [])
                    
                    if masks is not None and len(masks) > 0:
                        # Convert to tensor
                        if isinstance(masks, torch.Tensor):
                            masks_tensor = masks
                        else:
                            masks_tensor = torch.tensor(masks)
                        
                        # For binary masks, just convert to float; for logits, apply sigmoid
                        if use_binary:
                            masks_prob = masks_tensor.float()
                        else:
                            # Sigmoid to convert logits to probabilities
                            masks_prob = torch.sigmoid(masks_tensor.float())
                        
                        # Store per-object masks for multi_mask output
                        for i, oid in enumerate(obj_ids):
                            oid = int(oid)
                            detected_obj_ids.add(oid)
                            if i < masks_prob.shape[0]:
                                obj_mask = masks_prob[i].squeeze()
                                # Resize to original frame size
                                if obj_mask.shape != (frame_h, frame_w):
                                    obj_mask = F.interpolate(
                                        obj_mask.unsqueeze(0).unsqueeze(0).float(),
                                        size=(frame_h, frame_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze()
                                # Apply threshold and store
                                obj_mask = (obj_mask > threshold).to(torch.uint8)
                                masks_per_frame_per_obj[frame_idx][oid] = obj_mask.cpu()

                        # Combine masks for main output
                        if masks_prob.dim() > 2:
                            combined = masks_prob.max(dim=0)[0]
                        else:
                            combined = masks_prob
                        
                        combined = combined.squeeze()
                        if combined.shape != (frame_h, frame_w):
                            combined = F.interpolate(
                                combined.unsqueeze(0).unsqueeze(0).float(),
                                size=(frame_h, frame_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        
                        combined = (combined > threshold).float()
                        output_masks.append(combined.cpu())
                    else:
                        output_masks.append(torch.zeros((frame_h, frame_w), dtype=torch.float32))
                    
                    # Collect boxes
                    if "out_boxes_xywh" in frame_out:
                        boxes = frame_out["out_boxes_xywh"]
                        if boxes is not None and len(boxes) > 0:
                            if hasattr(boxes, 'tolist'):
                                all_boxes.extend(boxes.tolist())
                            else:
                                all_boxes.extend(boxes)
                else:
                    output_masks.append(torch.zeros((frame_h, frame_w), dtype=torch.float32))
            
            # Stack masks
            if output_masks:
                output_tensor = torch.stack(output_masks, dim=0)
            else:
                output_tensor = torch.zeros((num_frames, frame_h, frame_w), dtype=torch.float32)
            
            detected_ids_str = ",".join(str(oid) for oid in sorted(detected_obj_ids))
            
            # Build SAM3_MULTI_MASK output (compatible with Sam3ExtractObjectMask)
            multi_mask_output = {
                "masks_per_frame_per_obj": masks_per_frame_per_obj,
                "object_ids": sorted(detected_obj_ids),
                "num_frames": num_frames,
                "frame_size": (frame_h, frame_w),
            }
            
            # Close session
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            
            print(f"[SAM3 Native] Done! Output shape: {output_tensor.shape}")
            
        finally:
            if 'temp_dir' in dir() and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not keep_model_loaded:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (output_tensor, all_boxes if all_boxes else None, detected_ids_str, multi_mask_output)


class NativeSam3VideoMaskWithPrompts:
    """
    Segment and track objects in video using point/box prompts.
    Uses Facebook's native SAM3 library for maximum performance.
    
    Accepts SAM3_OBJECT_PROMPTS from Sam3ObjectPrompt nodes.
    Outputs SAM3_MULTI_MASK compatible with Sam3ExtractObjectMask.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_native_model": ("SAM3NATIVEMODEL",),
                "video_frames": ("IMAGE",),
            },
            "optional": {
                "object_prompts": ("SAM3_OBJECT_PROMPTS",),
                "multi_prompts": ("SAM3_MULTI_PROMPTS",),  # From SAM3MultiRegionCollector
                "positive_points": ("POINTS",),
                "negative_points": ("POINTS",),
                "bboxes": ("BBOX",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "propagation_direction": (["both", "forward", "backward"], {"default": "both"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MASK", "BBOX", "STRING", "SAM3_MULTI_MASK")
    RETURN_NAMES = ("masks", "bboxes", "detected_object_ids", "multi_mask")
    FUNCTION = "process_video"
    CATEGORY = "SAM3.1 Native/video"
    
    DESCRIPTION = """
    Segment objects in video using points/boxes with native SAM3 library.

    SAM 3.1 with Object Multiplex provides ~7x faster multi-object tracking.

    object_prompts: SAM3_OBJECT_PROMPTS from Sam3ObjectPrompt (multi-object)
    multi_prompts: SAM3_MULTI_PROMPTS from SAM3MultiRegionCollector (multi-region)
    positive_points: Click points on objects
    negative_points: Click points to exclude
    bboxes: Bounding boxes around objects
    propagation_direction: both (forward+backward), forward, or backward from frame_index

    Outputs:
    - masks: Combined mask of all objects
    - bboxes: Detected bounding boxes
    - detected_object_ids: Comma-separated IDs
    - multi_mask: Per-object masks for Sam3ExtractObjectMask
    """
    
    def process_video(self, sam3_native_model: Dict, video_frames: torch.Tensor,
                      object_prompts=None, multi_prompts=None, positive_points=None, negative_points=None,
                      bboxes=None, frame_index: int = 0, propagation_direction: str = "both",
                      threshold: float = 0.5, keep_model_loaded: bool = True):

        # Check inference cache
        cache_key = _inference_cache.make_key(
            model_name=sam3_native_model.get("model_name", ""),
            video_frames=video_frames,
            object_prompts=str(object_prompts) if object_prompts else None,
            multi_prompts=str(multi_prompts) if multi_prompts else None,
            positive_points=str(positive_points) if positive_points else None,
            negative_points=str(negative_points) if negative_points else None,
            bboxes=str(bboxes) if bboxes else None,
            frame_index=frame_index, propagation_direction=propagation_direction,
            threshold=threshold,
        )
        cached = _inference_cache.get(cache_key)
        if cached is not None:
            print(f"[SAM3 Native] Cache HIT — returning cached result")
            return cached

        predictor = sam3_native_model["predictor"]
        device = sam3_native_model.get("device", "cuda")
        autocast_dtype = sam3_native_model.get("autocast_dtype", torch.bfloat16)
        use_autocast = device == "cuda" and autocast_dtype in [torch.float16, torch.bfloat16]
        offload_to_cpu = sam3_native_model.get("offload_to_cpu", True)
        memory_mode = sam3_native_model.get("memory_mode", "balanced")

        print(f"[SAM3 Native] Processing video with point/box prompts (memory_mode={memory_mode})")

        num_frames = video_frames.shape[0]
        frame_h, frame_w = video_frames.shape[1], video_frames.shape[2]
        print(f"[SAM3 Native] Video: {num_frames} frames, {frame_w}x{frame_h}")

        try:
            # Save frames to disk for SAM3's async frame loader, which loads
            # frames lazily one-at-a-time. This is essential for large videos (4000+
            # frames) — the PIL list path eagerly loads ALL frames into one ~30GB tensor.
            # Use torch uint8 conversion to avoid numpy float64 intermediate (~2x RAM).
            import tempfile, shutil
            import time as _time
            _t0 = _time.perf_counter()
            temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
            print(f"[SAM3 Native] Saving {num_frames} frames to {temp_dir}...")
            video_uint8 = video_frames.cpu().mul(255).clamp_(0, 255).byte()

            from concurrent.futures import ThreadPoolExecutor
            def _save_frame(args):
                idx, frame_np = args
                Image.fromarray(frame_np).save(
                    os.path.join(temp_dir, f"{idx:05d}.jpg"), quality=80
                )

            # Iterate frame-by-frame to avoid materializing full numpy array
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
                pool.map(_save_frame, ((i, video_uint8[i].numpy()) for i in range(num_frames)))
            _t1 = _time.perf_counter()
            print(f"[SAM3 Native] Saved {num_frames} frames in {_t1 - _t0:.1f}s")
            del video_uint8
            gc.collect()

            # Wrap all predictor calls with autocast (matching the working VideoMasking node)
            _amp_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype)
            _amp_ctx.__enter__()

            # start_session with folder path — SAM3 uses AsyncImageFrameLoader
            # to load frames lazily on-demand (no giant upfront allocation).
            print(f"[SAM3 Native] Starting tracking session...")
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_dir,
                    offload_video_to_cpu=True,
                )
            )
            session_id = response["session_id"]
            print(f"[SAM3 Native] Session started: {session_id}")

            # Conditionally apply CPU offload on the inference state.
            # In speed mode, keep everything on GPU for maximum throughput.
            # In balanced/low_memory mode, offload to CPU to prevent VRAM OOM.
            import time
            session_data = predictor._all_inference_states.get(session_id)
            inference_state = session_data["state"] if session_data else None
            if inference_state is not None and offload_to_cpu:
                inference_state["offload_state_to_cpu"] = True
                inference_state["storage_device"] = torch.device("cpu")
                print(f"[SAM3 Native] Patched inference state: storage_device=cpu")
            elif inference_state is not None:
                print(f"[SAM3 Native] Speed mode: inference state stays on GPU")
            # Wait for async frame loader to finish
            images_obj = inference_state.get("images") if inference_state else None
            max_wait = 300
            total_waited = 0
            if images_obj is not None and hasattr(images_obj, 'all_frames_loaded'):
                while not images_obj.all_frames_loaded and total_waited < max_wait:
                    time.sleep(0.5)
                    total_waited += 0.5
                loaded = getattr(images_obj, 'num_loaded_frames', '?')
                print(f"[SAM3 Native] Frame loading complete: {loaded}/{num_frames} frames ({total_waited:.1f}s)")
            else:
                wait_time = min(10.0, num_frames * 0.02)
                time.sleep(wait_time)
                print(f"[SAM3 Native] Waited {wait_time:.1f}s for frame loading (estimated)")

            # NOTE: Do NOT call reset_session here. It clears cached_frame_outputs
            # and feature_cache which are needed for SAM2 point-based tracking.
            # The session is already fresh from start_session.

            # Get actual frame count from session and clamp frame_index
            try:
                session_data = predictor._all_inference_states.get(session_id)
                if session_data:
                    model_num_frames = session_data["state"].get("num_frames", num_frames)
                    if frame_index >= model_num_frames:
                        print(f"[SAM3 Native] WARNING: frame_index {frame_index} >= model frames {model_num_frames}, clamping to {model_num_frames - 1}")
                        frame_index = model_num_frames - 1
            except Exception:
                pass

            # Pre-seed cached_frame_outputs BEFORE add_prompt.
            # Required: _build_sam2_output returns {} for frames not in cache.
            try:
                session_data = predictor._all_inference_states.get(session_id)
                if session_data:
                    inference_state = session_data["state"]
                    actual_frames = inference_state.get("num_frames", num_frames)
                    cache = inference_state.setdefault("cached_frame_outputs", {})
                    seeded = 0
                    for i in range(actual_frames):
                        if i not in cache:
                            cache[i] = {}
                            seeded += 1
                    print(f"[SAM3 Native] Pre-seeded {seeded} frames (model has {actual_frames} frames)")
            except Exception as e:
                print(f"[SAM3 Native] Warning: Could not pre-seed cache: {e}")

            # =================================================================
            # COLLECT ALL PROMPTS FROM ALL SOURCES
            # Combines: object_prompts + multi_prompts + direct points/bboxes
            # All inputs are converted to POINTS for the SAM2 tracker path.
            # The SAM3.1 multiplex model routes point prompts through SAM2,
            # which handles propagation correctly. Boxes are converted to
            # corner points (labels 2=top-left, 3=bottom-right).
            # =================================================================
            native_prompts = []
            current_obj_id = 1

            # --- Source 1: object_prompts (from Sam3ObjectPrompt nodes) ---
            if object_prompts:
                obj_prompts = normalize_object_prompts_for_native(
                    object_prompts, frame_w, frame_h
                )
                for p in obj_prompts:
                    p["object_id"] = current_obj_id
                    # Convert box to corner points so everything uses SAM2 path
                    if p["box"] is not None:
                        bx, by, bw, bh = p["box"]
                        p["points"].append([bx, by])
                        p["labels"].append(2)  # top-left corner
                        p["points"].append([bx + bw, by + bh])
                        p["labels"].append(3)  # bottom-right corner
                        p["box"] = None
                    native_prompts.append(p)
                    current_obj_id += 1

            # --- Source 2: multi_prompts (from SAM3MultiRegionCollector) ---
            if multi_prompts:
                multi_p = normalize_multi_prompts_for_native(
                    multi_prompts, frame_w, frame_h
                )
                for p in multi_p:
                    p["object_id"] = current_obj_id
                    # Convert box to corner points
                    if p["box"] is not None:
                        bx, by, bw, bh = p["box"]
                        p["points"].append([bx, by])
                        p["labels"].append(2)
                        p["points"].append([bx + bw, by + bh])
                        p["labels"].append(3)
                        p["box"] = None
                    native_prompts.append(p)
                    current_obj_id += 1

            # --- Source 3: direct positive_points/negative_points/bboxes ---
            if bboxes is not None:
                print(f"[SAM3 Native] Raw bboxes input: {bboxes}")
            pos_pts = self._normalize_points_to_relative(positive_points, frame_w, frame_h)
            neg_pts = self._normalize_points_to_relative(negative_points, frame_w, frame_h)
            rel_boxes = self._normalize_bboxes_to_relative(bboxes, frame_w, frame_h)
            if rel_boxes:
                print(f"[SAM3 Native] Normalized bboxes (relative): {rel_boxes}")

            if pos_pts or neg_pts or rel_boxes:
                # Combine direct points + first box into one object
                direct_points = []
                direct_labels = []
                for pt in pos_pts:
                    direct_points.append(pt)
                    direct_labels.append(1)
                for pt in neg_pts:
                    direct_points.append(pt)
                    direct_labels.append(0)

                if rel_boxes:
                    # Convert first box to corner points
                    box = rel_boxes[0]
                    direct_points.append([box[0], box[1]])
                    direct_labels.append(2)  # top-left
                    direct_points.append([box[2], box[3]])
                    direct_labels.append(3)  # bottom-right

                if direct_points:
                    native_prompts.append({
                        "object_id": current_obj_id,
                        "points": direct_points,
                        "labels": direct_labels,
                        "box": None,
                    })
                    current_obj_id += 1

                # Additional bboxes become separate objects (as corner points)
                for box in rel_boxes[1:]:
                    native_prompts.append({
                        "object_id": current_obj_id,
                        "points": [[box[0], box[1]], [box[2], box[3]]],
                        "labels": [2, 3],
                        "box": None,
                    })
                    current_obj_id += 1

            added_obj_ids = set()

            print(f"[SAM3 Native] Processing {len(native_prompts)} prompts from all sources...")

            for prompt in native_prompts:
                obj_id = prompt["object_id"]
                has_points = bool(prompt["points"])

                print(f"[SAM3 Native]   Object {obj_id}: {len(prompt.get('points', []))} points, labels={prompt.get('labels', [])}")

                if not has_points:
                    print(f"[SAM3 Native]   Skipping - no points")
                    continue

                # All prompts use points (boxes converted to corner points above)
                # This ensures we go through the SAM2 tracker path in the multiplex model
                # Explicit float32 ensures no dtype mismatch from leaked autocast state
                request = dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    obj_id=obj_id,
                    points=torch.tensor(prompt["points"], dtype=torch.float32).cpu(),
                    point_labels=torch.tensor(prompt["labels"], dtype=torch.int32).cpu(),
                )

                print(f"[SAM3 Native]   Sending add_prompt for obj_id={obj_id}, points={prompt['points']}, labels={prompt['labels']}")
                response = predictor.handle_request(request=request)
                out = response.get("outputs", {}) if response else {}

                # The multiplex model's SAM2 path may return empty out_obj_ids
                # on add_prompt (cached_frame_outputs not yet populated).
                # This is expected - the object IS registered internally.
                # We track it and rely on propagation to produce masks.
                added_obj_ids.add(obj_id)
                out_obj_ids = out.get("out_obj_ids", []) if out else []
                if len(out_obj_ids) > 0:
                    print(f"[SAM3 Native]   OK - initial mask for IDs: {[int(x) for x in out_obj_ids]}")
                else:
                    print(f"[SAM3 Native]   Object {obj_id} registered (mask will appear after propagation)")

            print(f"[SAM3 Native] Added objects: {sorted(added_obj_ids)}")

            if not added_obj_ids:
                print(f"[SAM3 Native] No objects added - returning empty masks")
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
                empty_mask = torch.zeros((num_frames, frame_h, frame_w), dtype=torch.float32)
                return (empty_mask, None, "", {"masks_per_frame_per_obj": {}, "object_ids": [], "num_frames": num_frames, "frame_size": (frame_h, frame_w)})

            # NOTE: Do NOT pre-seed cached_frame_outputs. Pre-seeding all frames
            # with empty dicts breaks backward propagation and non-zero frame_index
            # (masks appear only on the prompt frame). The SAM3.1 propagation
            # populates the cache as it processes frames.

            # Auto-fix direction at video boundaries: "both" from the last frame
            # breaks because forward has nowhere to go; same for first frame backward.
            if propagation_direction == "both":
                if frame_index >= num_frames - 1:
                    propagation_direction = "backward"
                    print(f"[SAM3 Native] Auto-switched direction to 'backward' (prompt on last frame)")
                elif frame_index == 0:
                    propagation_direction = "forward"
                    print(f"[SAM3 Native] Auto-switched direction to 'forward' (prompt on first frame)")

            # Propagate masks through video
            # Pass start_frame_index so _get_processing_order doesn't require
            # previous_stages_out (which is only populated by text/detection prompts)
            # Capture the last add_prompt response so we can fill the prompt
            # frame if propagation skips it (backward mode excludes start frame)
            last_add_prompt_out = out

            print(f"[SAM3 Native] Propagating masks through video from frame {frame_index} ({propagation_direction})...")
            outputs_per_frame = {}
            detected_obj_ids = set()
            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    start_frame_index=frame_index,
                    propagation_direction=propagation_direction,
                )
            ):
                fidx = response["frame_index"]
                fout = response["outputs"]
                # In balanced/low_memory mode, move mask tensors to CPU immediately
                # to avoid VRAM accumulation over thousands of frames.
                # In speed mode, keep on GPU to avoid transfer overhead.
                if offload_to_cpu:
                    for _k in ("out_mask_logits", "out_binary_masks"):
                        if _k in fout and isinstance(fout[_k], torch.Tensor):
                            fout[_k] = fout[_k].cpu()
                outputs_per_frame[fidx] = fout
                if fidx in (0, 1, frame_index, frame_index - 1):
                    f_oids = fout.get("out_obj_ids", [])
                    f_masks = fout.get("out_binary_masks", fout.get("out_mask_logits", None))
                    n_m = len(f_masks) if f_masks is not None else 0
                    print(f"[SAM3 Native]   Frame {fidx}: obj_ids={[int(x) for x in f_oids]}, n_masks={n_m}")

            # If the prompt frame wasn't included in propagation output (happens
            # in backward mode where processing_order starts at start_frame-1),
            # fill it from the add_prompt response so masks aren't dropped.
            if frame_index not in outputs_per_frame and last_add_prompt_out:
                outputs_per_frame[frame_index] = last_add_prompt_out
                print(f"[SAM3 Native]   Frame {frame_index} (prompt frame): filled from add_prompt response")

            print(f"[SAM3 Native] Propagation complete, got {len(outputs_per_frame)} frames")

            _amp_ctx.__exit__(None, None, None)

            # Build output masks
            output_masks = []
            all_boxes = []
            masks_per_frame_per_obj = {}

            for frame_idx in range(num_frames):
                masks_per_frame_per_obj[frame_idx] = {}
                
                if frame_idx in outputs_per_frame:
                    frame_out = outputs_per_frame[frame_idx]
                    
                    # Try out_mask_logits first, fall back to out_binary_masks
                    masks = frame_out.get("out_mask_logits", None)
                    use_binary = False
                    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
                        masks = frame_out.get("out_binary_masks", None)
                        use_binary = True
                    
                    obj_ids = frame_out.get("out_obj_ids", [])
                    
                    if masks is not None and len(masks) > 0:
                        if isinstance(masks, torch.Tensor):
                            masks_tensor = masks
                        else:
                            masks_tensor = torch.tensor(masks)
                        
                        if use_binary:
                            masks_prob = masks_tensor.float()
                        else:
                            masks_prob = torch.sigmoid(masks_tensor.float())
                        
                        for i, oid in enumerate(obj_ids):
                            oid = int(oid)
                            detected_obj_ids.add(oid)
                            if i < masks_prob.shape[0]:
                                obj_mask = masks_prob[i].squeeze()
                                if obj_mask.shape != (frame_h, frame_w):
                                    obj_mask = F.interpolate(
                                        obj_mask.unsqueeze(0).unsqueeze(0).float(),
                                        size=(frame_h, frame_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze()
                                obj_mask = (obj_mask > threshold).to(torch.uint8)
                                masks_per_frame_per_obj[frame_idx][oid] = obj_mask.cpu()

                        if masks_prob.dim() > 2:
                            combined = masks_prob.max(dim=0)[0]
                        else:
                            combined = masks_prob
                        combined = combined.squeeze()
                        if combined.shape != (frame_h, frame_w):
                            combined = F.interpolate(
                                combined.unsqueeze(0).unsqueeze(0).float(),
                                size=(frame_h, frame_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        combined = (combined > threshold).float()
                        output_masks.append(combined.cpu())
                    else:
                        output_masks.append(torch.zeros((frame_h, frame_w), dtype=torch.float32))
                    
                    if "out_boxes_xywh" in frame_out:
                        boxes_out = frame_out["out_boxes_xywh"]
                        if boxes_out is not None and len(boxes_out) > 0:
                            if hasattr(boxes_out, 'tolist'):
                                all_boxes.extend(boxes_out.tolist())
                            else:
                                all_boxes.extend(boxes_out)
                else:
                    output_masks.append(torch.zeros((frame_h, frame_w), dtype=torch.float32))
            
            if output_masks:
                output_tensor = torch.stack(output_masks, dim=0)
            else:
                output_tensor = torch.zeros((num_frames, frame_h, frame_w), dtype=torch.float32)
            
            detected_ids_str = ",".join(str(oid) for oid in sorted(detected_obj_ids))
            
            multi_mask_output = {
                "masks_per_frame_per_obj": masks_per_frame_per_obj,
                "object_ids": sorted(detected_obj_ids),
                "num_frames": num_frames,
                "frame_size": (frame_h, frame_w),
            }
            
            # close_session — frees inference_state, cached frames, feature cache
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

            # Free propagation outputs now that masks are built
            del outputs_per_frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[SAM3 Native] Done! Output shape: {output_tensor.shape}")

        finally:
            if 'temp_dir' in dir() and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not keep_model_loaded:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        result = (output_tensor, all_boxes if all_boxes else None, detected_ids_str, multi_mask_output)
        _inference_cache.put(cache_key, result)
        return result

    def _normalize_points_to_relative(self, points, width: int, height: int) -> List[List[float]]:
        """Convert points to relative coordinates (0-1 range)."""
        result = []
        if points is None:
            return result
        
        if isinstance(points, dict):
            pts = points.get("points", [])
            for pt in pts:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                    # Check if already pixel coords
                    if x > 1 or y > 1:
                        x = x / width
                        y = y / height
                    result.append([x, y])
        elif isinstance(points, (list, tuple)):
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                    if x > 1 or y > 1:
                        x = x / width
                        y = y / height
                    result.append([x, y])
        
        return result
    
    def _normalize_bboxes_to_relative(self, bboxes, width: int, height: int) -> List[List[float]]:
        """Convert bboxes to relative coordinates [x1, y1, x2, y2]."""
        result = []
        if bboxes is None:
            return result
        
        boxes = bboxes if isinstance(bboxes, list) else [bboxes]
        
        for box in boxes:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                # Check if already pixel coords
                if any(v > 1 for v in [x1, y1, x2, y2]):
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                result.append([x1, y1, x2, y2])
        
        return result


class UnloadNativeSAM3Model:
    """
    Unload Native SAM3 model from GPU memory.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_native_model": ("SAM3NATIVEMODEL",),
            },
            "optional": {
                "input_masks": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "unload"
    CATEGORY = "SAM3.1 Native/loaders"
    OUTPUT_NODE = True
    
    DESCRIPTION = """
    Unload Native SAM3 model from GPU memory.
    Connect after SAM3 processing to free VRAM.
    """
    
    def unload(self, sam3_native_model: Dict, input_masks: torch.Tensor = None):
        print("[SAM3 Native] Unloading model...")
        
        try:
            predictor = sam3_native_model.get("predictor")
            if predictor is not None:
                try:
                    predictor.handle_request(request=dict(type="cleanup"))
                except:
                    pass
                
                del predictor
                sam3_native_model["predictor"] = None
        except Exception as e:
            print(f"[SAM3 Native] Warning during unload: {e}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        mm.soft_empty_cache()
        
        print("[SAM3 Native] Model unloaded")
        
        if input_masks is not None:
            return (input_masks,)
        return (torch.zeros((1, 64, 64), dtype=torch.float32),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadNativeSAM3VideoModel": LoadNativeSAM3VideoModel,
    "NativeSam3VideoMask": NativeSam3VideoMask,
    "NativeSam3VideoMaskWithPrompts": NativeSam3VideoMaskWithPrompts,
    "UnloadNativeSAM3Model": UnloadNativeSAM3Model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNativeSAM3VideoModel": "SAM3 Native Model Loader (3.1 Multiplex)",
    "NativeSam3VideoMask": "SAM3 Native Video Mask (Text)",
    "NativeSam3VideoMaskWithPrompts": "SAM3 Native Video Mask (Points/Boxes)",
    "UnloadNativeSAM3Model": "SAM3 Native Unload Model",
}
