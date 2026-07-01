import json
import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Response
import tempfile
import os
from app.services.memory_manager import memory_manager

from app.services.synthesis_llm import generate_final_report
from app.utils.image_compositor import create_final_ultrasound_image

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/synthesis",
    tags=["Synthesis Node"]
)

# Define media directory for temporary and final images
MEDIA_DIR = Path("media/synthesis")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/review/{session_id}")
async def synthesis_review(session_id: str):
    """
    Synthesis Node Endpoint (Multi-Module).
    Cross-references clinical, ultrasound, and FNAC data from the session state
    to generate a final medical report.
    """
    try:
        session_ctx = await memory_manager.load_context(session_id)
        if not session_ctx:
            raise HTTPException(status_code=404, detail="Session not found")

        clinical_data = session_ctx.diagnostic_context.get("clinical", {})
        ultrasound_data = session_ctx.diagnostic_context.get("ultrasound", {})
        fnac_data = session_ctx.diagnostic_context.get("fnac", {})

        if not clinical_data and not ultrasound_data and not fnac_data:
            raise HTTPException(status_code=400, detail="No diagnostic data found for this session")

        # Call LLM Service with multi-module data
        try:
            from app.services.synthesis_llm import generate_final_report
            final_report = await generate_final_report(
                clinical_data=clinical_data,
                ultrasound_data=ultrasound_data,
                fnac_data=fnac_data
            )
        except Exception as e:
            logger.error(f"Synthesis LLM failed: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate final report")

        response_data = final_report.model_dump()
        response_data["session_id"] = session_id

        # Persist to memory
        await memory_manager.save_diagnostic(
            session_id=session_id,
            node_type="synthesis",
            data=response_data,
        )

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in synthesis review: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/image/{image_id}")
async def get_synthesis_image(image_id: int):
    """
    Retrieves a diagnostic image from the database.
    """
    image_data = await memory_manager.get_image(image_id)
    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image_data, media_type="image/png")

