FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY ./ai-retinal-screening-system /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] pillow numpy pandas opencv-python-headless \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

