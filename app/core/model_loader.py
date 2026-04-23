"""
Lean model loader.

Models are cached in memory via @functools.lru_cache after the first
load to prevent disk I/O latency on subsequent requests, prioritizing
SPEED over memory footprint constraints.
"""

import logging
from pathlib import Path
import functools

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOCAL_MODEL_PATH = BASE_DIR / "models" / "compressed" / "disease_compressed.joblib"

@functools.lru_cache(maxsize=None)
def load_production_model(model_name: str = "ThyraX_Disease_Classifier"):
    """
    Fetch the production model directly from the local joblib file
    and cache it in memory.

    Args:
        model_name: Identifier for logging purposes.

    Returns:
        Loaded scikit-learn / XGBoost model ready for .predict().

    Raises:
        RuntimeError: If local fallback fails.
    """
    try:
        from joblib import load as joblib_load

        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Local model file not found at {LOCAL_MODEL_PATH}"
            )

        model = joblib_load(str(LOCAL_MODEL_PATH))
        logger.info(f"✅ Successfully loaded local model '{model_name}' from {LOCAL_MODEL_PATH} into memory cache")
        return model

    except Exception as local_error:
        logger.error(f"❌ Local load failed: {local_error}")
        raise RuntimeError(
            f"Cannot load disease model. Local fallback at {LOCAL_MODEL_PATH} failed: "
            f"{local_error}"
        )
