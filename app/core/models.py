"""
High-Performance Model Loading.

Models are cached in memory via @functools.lru_cache after the first
load to prevent disk I/O latency on subsequent requests, prioritizing
SPEED over memory footprint constraints.
"""

import logging
from pathlib import Path
import functools

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=None)
def load_segmentation_model():
    """Load and return a cached segmentation ONNX model session."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = BASE_DIR / "models" / "compressed" / "segmentation.onnx"
    logger.info(f"Loading segmentation ONNX model from {model_path} into memory cache")
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


@functools.lru_cache(maxsize=None)
def load_classification_model():
    """Load and return the cached 'efficientnet_b4_medical_final.onnx' model session.
    
    This is the high-accuracy model requested by the user.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = BASE_DIR / "models" / "compressed" / "efficientnet_b4_medical_final.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Classification model not found at {model_path}")
        
    logger.info(f"Loading classification ONNX model from {model_path} into memory cache")
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


@functools.lru_cache(maxsize=None)
def load_gatekeeper_model():
    """Load and return a cached MobileNetV2 gatekeeper ONNX session.

    Classes: 0 = 'other', 1 = 'ultrasound'.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = BASE_DIR / "models" / "compressed" / "gatekeeper.onnx"
    logger.info(f"Loading gatekeeper ONNX model from {model_path} into memory cache")
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


@functools.lru_cache(maxsize=None)
def load_fnac_gatekeeper_model():
    """Load and return a cached FNAC gatekeeper ONNX session.

    Classes: 0 = 'other', 1 = 'fnac'.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = BASE_DIR / "models" / "compressed" / "fnac_gatekeeper.onnx"
    if not model_path.exists():
        # Handle graceful degradation if not deployed yet
        raise FileNotFoundError(f"FNAC gatekeeper model not found at {model_path}.")
    logger.info(f"Loading FNAC gatekeeper ONNX model from {model_path} into memory cache")
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


@functools.lru_cache(maxsize=None)
def load_fnac_model():
    """Load and return a cached FNAC Bethesda classifier ONNX session.

    Classes: 0–5 mapping to Bethesda I–VI.

    Raises:
        FileNotFoundError: If the FNAC model has not been deployed yet.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = BASE_DIR / "models" / "compressed" / "fnac_bethesda.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"FNAC model not found at {model_path}. "
            "Train and export the model first."
        )
    logger.info(f"Loading FNAC ONNX model from {model_path} into memory cache")
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
