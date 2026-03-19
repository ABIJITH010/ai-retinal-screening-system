from pydantic import BaseModel
from datetime import datetime

class ReportCreate(BaseModel):
    patient_id: int
    image_url: str
    heatmap_url: str
    prediction: str
    confidence: float

class ReportUpdate(BaseModel):
    prediction: str
    confidence: float

class ReportResponse(BaseModel):
    id: int
    patient_id: int
    image_url: str
    heatmap_url: str
    prediction: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True
