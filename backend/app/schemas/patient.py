from pydantic import BaseModel
from datetime import datetime

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    address: str

class PatientUpdate(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    address: str

class PatientResponse(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    phone: str
    address: str
    created_at: datetime
    doctor_id: int

    class Config:
        from_attributes = True
