import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool

from app.core.security import verify_internal_api_key
from app.services.document_service import extract_text_from_image

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/document",
    tags=["Document OCR"],
    dependencies=[Depends(verify_internal_api_key)],
)

@router.post("/ocr")
async def document_ocr(file: UploadFile = File(...)):
    """
    Local, memory-efficient OCR endpoint to extract text from medical documents
    and prescriptions using pytesseract and OpenCV.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        # Just a warning or soft check, in case client sends octet-stream
        pass

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Run preprocessing and OCR logic inside run_in_threadpool
        text = await run_in_threadpool(extract_text_from_image, image_bytes)
        return {"status": "success", "extracted_text": text}
    except RuntimeError as e:
        logger.error(f"OCR Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        logger.error(f"OCR Value Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected OCR error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during OCR processing.")
