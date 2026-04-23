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


async def run_segmentation_inference(session, input_name, input_data):
    """
    Run ONNX segmentation inference in threadpool.

    Args:
        session: ONNX InferenceSession
        input_name: Input node name
        input_data: Preprocessed input array

    Returns:
        Model output
    """
    def _inference():
        return session.run(None, {input_name: input_data})

    return await run_in_threadpool(_inference)


async def run_classification_inference(session, input_name, input_data):
    """
    Run ONNX classification inference in threadpool.

    Args:
        session: ONNX InferenceSession
        input_name: Input node name
        input_data: Preprocessed input array

    Returns:
        Model output (class probabilities)
    """
    def _inference():
        return session.run(None, {input_name: input_data})

    return await run_in_threadpool(_inference)
