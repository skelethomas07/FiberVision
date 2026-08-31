FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt requirements.txt ./
COPY vendor/sem_fiber_ai/requirements.txt ./vendor/sem_fiber_ai/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY vendor ./vendor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
