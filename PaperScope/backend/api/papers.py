from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database import get_db
from backend.models.paper import Paper
from pydantic import BaseModel, ConfigDict
from datetime import datetime

router = APIRouter(prefix="/papers", tags=["papers"])

# Pydantic Schemas for response validation
class PaperResponse(BaseModel):
    id: int
    title: str
    authors: str
    source: str
    published_date: datetime
    url: str
    abstract: Optional[str]
    ai_summary: Optional[str]
    matched_keywords: Optional[str]

    model_config = ConfigDict(from_attributes=True)

@router.get("/", response_model=List[PaperResponse])
def get_papers(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Fetch the latest crawled papers from the database."""
    papers = db.query(Paper).order_by(Paper.published_date.desc()).offset(skip).limit(limit).all()
    return papers
