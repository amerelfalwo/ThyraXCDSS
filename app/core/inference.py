"""
Inference helpers for CPU-bound model operations.

Wraps synchronous inference calls in async-friendly threadpool
to prevent blocking the FastAPI event loop.
"""

from fastapi.concurrency import run_in_threadpool
import logging

logger = logging.getLogger(__name__)


async def run_clinical_inference(model, df):
    """
    Run XGBoost clinical model prediction in threadpool.

    Args:
        model: Loaded XGBoost model
        df: Pandas DataFrame with features in correct order

    Returns:
        Tuple of (prediction, probabilities)
    """
    def _inference():
        pred = int(model.predict(df)[0])
        probs = model.predict_proba(df)[0]
        return pred, probs

    return await run_in_threadpool(_inference)

