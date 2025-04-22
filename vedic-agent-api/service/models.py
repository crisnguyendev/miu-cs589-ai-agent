from pydantic import BaseModel
from typing import List, Dict

class RetrievalResult(BaseModel):
    verse: str
    source: str
    score: float

class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]

class FeedbackRequest(BaseModel):
    query: str
    verse_id: str
    is_relevant: bool