from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import health, auth, patients, reports
from app.db.database import engine, Base
from pathlib import Path

app = FastAPI()

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Mount uploads directory as static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend running 🚀"}

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(reports.router)

Base.metadata.create_all(bind=engine)
