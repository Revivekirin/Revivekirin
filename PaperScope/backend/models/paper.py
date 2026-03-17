from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime
from backend.database import Base

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    authors = Column(String, nullable=False) # Store as comma-separated or JSON string
    source = Column(String, index=True, nullable=False) # e.g., 'arXiv', 'NeurIPS'
    published_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    url = Column(String, unique=True, index=True, nullable=False)
    
    # We will NOT store the PDF. We only store abstract and AI summary
    abstract = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True) # Populated later by the Summarizer
    
    matched_keywords = Column(String, nullable=True) # e.g., 'Agentic AI, LLM'
    
    created_at = Column(DateTime, default=datetime.utcnow)
