cat > /workspace/ComfyUI/custom_nodes/ComfyUI-SAM3-1/__init__.py << 'EOF'
"""
ComfyUI-SAM3.1
==================
Native SAM3/SAM3.1 video masking nodes using Facebook's sam3 library.
Supports Object Multiplex for ~7x faster multi-object tracking.
"""
import os
import sys
import subprocess

def _auto_install():
    node_dir = os.path.dirname(os.path.abspath(__file__))
    sam3_src = os.path.join(node_dir, "sam3_src", "sam3")
    
    # Check if our patched sam3 is already installed
    try:
        import sam3
        sam3_path = getattr(sam3, '__file__', '')
        if 'sam3_src' in sam3_path or 'sam3_patched' in sam3_path:
            return  # Already using patched version
    except ImportError:
        pass
    
    # Install setuptools<70 first (required for pkg_resources on Python 3.12)
    print("[SAM3.1] Installing setuptools<70...")
    subprocess.run([sys.executable, "-m", "pip", "install", "setuptools<70", "-q"], check=False)
    
    # Install patched sam3 from sam3_src
    sam3_src_dir = os.path.join(node_dir, "sam3_src")
    if os.path.exists(sam3_src_dir):
        print("[SAM3.1] Installing patched sam3 library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", sam3_src_dir, "--no-deps", "-q"], check=False)
        print("[SAM3.1] Patched sam3 installed successfully.")
    else:
        print("[SAM3.1] WARNING: sam3_src not found! Please run install.sh manually.")
    
    # Rename conflicting bundled sam3 in comfyui-rmbg if present
    comfyui_dir = os.path.dirname(node_dir)
    rmbg_sam3 = os.path.join(comfyui_dir, "comfyui-rmbg", "models", "sam3")
    if os.path.exists(rmbg_sam3):
        print("[SAM3.1] Renaming conflicting comfyui-rmbg/models/sam3...")
        os.rename(rmbg_sam3, rmbg_sam3 + "_backup")

_auto_install()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
EOF
echo "Done!"