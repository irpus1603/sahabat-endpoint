# device_select.py
import os
import sys
from pathlib import Path

# Global singletons
det_model = None
label_map = None
device_hint = None  # 'cuda:0' | 'mps' | 'cpu'

# ---- Configuration: point these to your actual files ----
MODEL_DIR = Path("models")
MODEL_PT = MODEL_DIR / "yolo11n.pt"           # PyTorch model for CUDA
MODEL_MLpackage = MODEL_DIR / "yolo11n.mlpackage"  # CoreML (MPS build)
MODEL_ONNX = MODEL_DIR / "yolov8n.onnx"       # ONNX fallback
# --------------------------------------------------------

def _set_safe_env():
    """
    Set env vars once to avoid fork/init crashes & tame BLAS threads.
    Call early before importing heavy libs.
    """
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _has_nvidia_cuda():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False

def _has_mps():
    # Apple Metal backend via PyTorch availability.
    try:
        import torch
        return (sys.platform == "darwin"
                and getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available())
    except Exception:
        return False

def _coreml_ready():
    # Prefer CoreML .mlpackage for MPS “build”
    return (sys.platform == "darwin" and MODEL_MLpackage.exists())

def _onnx_ready():
    # We can run .onnx if file exists and onnxruntime is installed
    if not MODEL_ONNX.exists():
        return False
    try:
        import onnxruntime as ort  # noqa: F401
        return True
    except Exception:
        return False

def _choose_model_and_device():
    """
    Decide which file + device string to use for Ultralytics YOLO().
    Returns (model_path: Path, device_str: str)
    """
    # 1) NVIDIA CUDA -> PyTorch .pt on 'cuda:0'
    if _has_nvidia_cuda() and MODEL_PT.exists():
        return (MODEL_PT, "cuda:0")

    # 2) Apple MPS -> CoreML .mlpackage (Ultralytics supports CoreML artifact)
    if _has_mps() and _coreml_ready():
        # Ultralytics will route CoreML models appropriately; device hint 'mps' is fine
        return (MODEL_MLpackage, "mps")

    # 3) Otherwise -> ONNX on CPU
    if _onnx_ready():
        return (MODEL_ONNX, "cpu")

    # 4) Last resort: if only .pt exists, run it on CPU
    if MODEL_PT.exists():
        return (MODEL_PT, "cpu")

    # If nothing is present, raise a clear error
    raise FileNotFoundError(
        f"No suitable model found in {MODEL_DIR}. "
        f"Looked for: {MODEL_PT.name}, {MODEL_MLpackage.name}, {MODEL_ONNX.name}"
    )

def get_detector():
    """
    Lazy-load the best available detector and expose (det_model, label_map, device_hint).
    """
    global det_model, label_map, device_hint
    if det_model is not None:
        return det_model, label_map, device_hint

    _set_safe_env()

    # Import here (after env set) to avoid fork/init issues on macOS
    from ultralytics import YOLO

    model_path, device = _choose_model_and_device()
    device_hint = device

    # Create the model
    det_model = YOLO(str(model_path), task="detect")

    # Names/labels map (works across PT/CoreML/ONNX exports if embedded)
    # If not present, build a fallback from model metadata where possible.
    label_map = getattr(det_model, "names", None)
    if not label_map:
        # Fallback: try model attributes
        try:
            label_map = det_model.model.names  # some exports use .model.names
        except Exception:
            # Default to empty dict -> caller can handle unknown labels as class IDs
            label_map = {}

    return det_model, label_map, device_hint
