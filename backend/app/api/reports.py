from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models.report import Report
from app.models.patient import Patient
from app.models.users import User
from app.schemas.report import ReportCreate, ReportResponse
from app.api.auth import get_current_user
import os
import shutil
from pathlib import Path

router = APIRouter(prefix="/api/reports", tags=["reports"])

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    try:
        file_path = UPLOAD_DIR / upload_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return str(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to save file: {str(e)}"
        )

@router.post("/", response_model=ReportResponse)
async def create_report(
    patient_id: int = Query(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new report with image upload and AI prediction
    
    - Verify patient belongs to doctor
    - Save uploaded image
    - Run AI prediction
    - Store report in database
    """
    # Verify patient belongs to current doctor
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found or access denied"
        )
    
    # Save uploaded image
    image_path = save_upload_file(file)
    
    # Run AI prediction
    try:
        from app.services.ai_service import predict_dr_stage
        prediction, confidence, heatmap_path = predict_dr_stage(image_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI prediction failed: {str(e)}"
        )
    
    # Create report record
    new_report = Report(
        patient_id=patient_id,
        image_url=image_path,
        heatmap_url=heatmap_path,
        prediction=prediction,
        confidence=confidence
    )
    
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return new_report

@router.get("/", response_model=list[ReportResponse])
def get_all_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all reports for the authenticated doctor's patients"""
    reports = db.query(Report).join(Patient).filter(
        Patient.doctor_id == current_user.id
    ).all()
    return reports

@router.get("/{report_id}", response_model=ReportResponse)
def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific report (only if it belongs to current doctor's patient)"""
    report = db.query(Report).join(Patient).filter(
        Report.id == report_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    return report

@router.get("/patient/{patient_id}", response_model=list[ReportResponse])
def get_patient_reports(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all reports for a specific patient (only if it belongs to current doctor)"""
    # Verify patient belongs to current doctor
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found or access denied"
        )
    
    reports = db.query(Report).filter(Report.patient_id == patient_id).all()
    return reports

@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a report (only if it belongs to current doctor's patient)"""
    report = db.query(Report).join(Patient).filter(
        Report.id == report_id,
        Patient.doctor_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Delete associated files
    if os.path.exists(report.image_url):
        os.remove(report.image_url)
    if os.path.exists(report.heatmap_url):
        os.remove(report.heatmap_url)
    
    db.delete(report)
    db.commit()
    
    return None
