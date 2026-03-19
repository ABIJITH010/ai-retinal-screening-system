from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime
from app.db.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    
    phone = Column(String)
    address = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    doctor_id = Column(Integer, ForeignKey("users.id"))
