"""
ComfyUI-SAM3.1
==================
Native SAM3/SAM3.1 video masking nodes using Facebook's sam3 library.
"""
import os
import sys
import subprocess
import importlib

def _auto_install():
    node_dir = os.path.dirname(os.path.abspath(__file__))
    sam3_src_dir = os.path.join(node_dir, "sam3_src")

    try:
        import sam3
        sam3_path = getattr(sam3, '__file__', '') or ''
        if 'sam3_src' in sam3_path or 'ComfyUI-SAM3' in sam3_path:
            print("[SAM3.1] Patched sam3 already loaded.")
            return
    except ImportError:
        pass

    print("[SAM3.1] Installing setuptools<70...")
    subprocess.run([sys.executable, "-m", "pip", "install", "setuptools<70", "-q"], check=False)

    if os.path.exists(sam3_src_dir):
        print("[SAM3.1] Installing patched sam3 library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", sam3_src_dir, "--no-deps", "-q"], check=False)
        importlib.invalidate_caches()
        if 'sam3' in sys.modules:
            del sys.modules['sam3']
        try:
            import sam3
            print(f"[SAM3.1] sam3 loaded from: {sam3.__file__}")
        except Exception as e:
            print(f"[SAM3.1] Warning: {e}")
    else:
        print("[SAM3.1] ERROR: sam3_src not found!")

    rmbg_sam3 = os.path.join(os.path.dirname(node_dir), "comfyui-rmbg", "models", "sam3")
    if os.path.exists(rmbg_sam3):
        os.rename(rmbg_sam3, rmbg_sam3 + "_backup")

_auto_install()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
