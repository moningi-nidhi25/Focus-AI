from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import current_user
from pydantic import BaseModel

from app.services.content_extractor import MAX_UPLOAD_BYTES, extract_and_normalize, extract_document

router = APIRouter(prefix="/content", tags=["Content Extraction"], dependencies=[Depends(current_user)])


class ExtractionRequest(BaseModel):
    url: Optional[str] = None
    html: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None


class ExtractionResponse(BaseModel):
    source_type: str
    url: Optional[str] = None
    title: str
    cleaned_text: str
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


@router.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
def extract_content(payload: ExtractionRequest):
    """Extract, clean and normalize text from a webpage URL, raw HTML, YouTube video, or text snippet."""
    return extract_and_normalize(url=payload.url, html=payload.html, text=payload.text, title=payload.title)


@router.post("/extract/file", response_model=ExtractionResponse)
async def extract_file(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF, DOCX, TXT or MD document."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return extract_document(file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=422, detail="The file could not be read. Is it corrupted or password protected?")
