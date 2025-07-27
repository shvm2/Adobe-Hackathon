FROM --platform=linux/amd64 python:3.9-slim
WORKDIR /app
# Install build-essential and poppler-utils (dependency for pdfplumber)
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY extract_outline.py .
# Create input and output directories as expected by the script
RUN mkdir -p /app/input /app/output
CMD ["python", "extract_outline.py"]
