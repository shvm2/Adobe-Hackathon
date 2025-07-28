FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install build-essential and poppler-utils (dependency for pdfplumber)
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# If using spaCy, download the small English model
# RUN python -m spacy download en_core_web_sm

COPY . .

# Create input and output directories as expected by the script
# Ensure the script can read input JSON from /app/input/ and PDFs from /app/input/PDFs/
RUN mkdir -p /app/input /app/output

CMD ["python", "main_processor.py"]