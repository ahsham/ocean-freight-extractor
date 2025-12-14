FROM python:3.11-slim

WORKDIR /app

# poppler-utils: pdfplumber + pdf2image rendering support
# tesseract-ocr: simple OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py extractors.py llm_providers.py negotiate.py ./

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
