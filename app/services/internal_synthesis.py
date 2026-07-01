"""
Internal Synthesis Service.

Orchestrates the synthesis node: collects clinical + ultrasound data,
calls the LLM reviewer, and stores the result.
No image compositing — purely data-driven.
"""

import logging
from typing import Optional
from app.services.synthesis_llm import generate_final_report

logger = logging.getLogger(__name__)


async def run_internal_synthesis(
    session_id: str,
    clinical_data: dict,
    ultrasound_data: dict,
    original_image_bytes: bytes = None,
    mask_image_bytes: bytes = None,
    bbox: list = None,
) -> Optional[dict]:
    """
    Runs the Synthesis Node: cross-references clinical labs with ultrasound
    classification results via LLM, then stores the final report.

    Image and mask parameters are accepted for API compatibility but NOT
    sent to the LLM — synthesis is purely numerical/textual.
    """
    try:
        logger.info(f"Triggering data-driven synthesis for session {session_id}")

        # 1. Call LLM — numbers only, no images
        final_report = await generate_final_report(
            clinical_data=clinical_data,
            ultrasound_data=ultrasound_data,
        )

        # 2. Build response
        response_data = final_report.model_dump()
        response_data["clinical_data_snapshot"] = clinical_data
        response_data["ultrasound_data_snapshot"] = {
            k: v for k, v in ultrasound_data.items()
            if k not in ("images", "segmentation_mask", "mask_bytes")
        }

        # 3. Persist to memory
        from app.services.memory_manager import memory_manager
        await memory_manager.save_diagnostic(
            session_id=session_id,
            node_type="synthesis",
            data=response_data,
        )

        logger.info(
            f"Synthesis saved for session {session_id}: "
            f"{final_report.corrected_classification} / {final_report.tumor_stage}"
        )
        return response_data

    except Exception as e:
        logger.error(f"Internal synthesis failed: {e}", exc_info=True)
        return None
