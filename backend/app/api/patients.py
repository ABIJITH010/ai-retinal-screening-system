from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models.patient import Patient
from app.models.users import User
from app.schemas.patient import PatientCreate, PatientUpdate, PatientResponse
from app.api.auth import get_current_user

router = APIRouter(prefix="/api/patients", tags=["patients"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=PatientResponse)
def create_patient(
    request: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new patient for the authenticated doctor"""
    new_patient = Patient(
        name=request.name,
        age=request.age,
        gender=request.gender,
        phone=request.phone,
        address=request.address,
        doctor_id=current_user.id
    )
    
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    
    return new_patient

@router.get("", response_model=list[PatientResponse])
def get_all_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all patients for the authenticated doctor"""
    patients = db.query(Patient).filter(Patient.doctor_id == current_user.id).all()
    return patients

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific patient by ID (only own patients)"""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int,
    request: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a patient's information (only own patients)"""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    patient.name = request.name
    patient.age = request.age
    patient.gender = request.gender
    patient.phone = request.phone
    patient.address = request.address
    
    db.commit()
    db.refresh(patient)
    
    return patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a patient (only own patients)"""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    db.delete(patient)
    db.commit()
    
    return None
