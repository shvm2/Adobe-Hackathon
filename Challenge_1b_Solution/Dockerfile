FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install build-essential and poppler-utils (dependency for pdfplumber)
# These are necessary for pdfplumber to function correctly for PDF text extraction.
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts
COPY pdf_extractor.py .
COPY process_challenge_1b.py .

# Create input and output directories as expected by the script
# These directories will be mounted as volumes when running the container.
RUN mkdir -p /app/input /app/output

# Command to run the main processing script for Challenge 1b
CMD ["python", "process_challenge_1b.py"]
