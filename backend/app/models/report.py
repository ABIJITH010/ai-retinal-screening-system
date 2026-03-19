from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from datetime import datetime
from app.db.database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    image_url = Column(String)
    heatmap_url = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
