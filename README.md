Adobe India Hackathon 2025 - Challenge 1a Solution: Understand Your Document
Overview

This project provides a robust and intelligent solution for Challenge 1a: Understand Your Document of the Adobe India Hackathon 2025. The primary mission is to extract a structured outline (document title and hierarchical headings like H1, H2, H3, and H4) from raw PDF documents and output them in a clean, hierarchical JSON format. This extracted outline serves as a foundational component for smarter document experiences.
Approach

The solution employs a multi-stage, hybrid approach to achieve highly accurate title and heading detection:

    Detailed PDF Content Extraction:

        Utilizes pdfplumber to meticulously parse each page of the input PDF.

        Extracts comprehensive layout features for every line of text, including font size, absolute and relative vertical/horizontal positions (y0, x0, left_margin), font variation (consistency), and word count. This rich feature set is critical for effective rule-based and machine learning analysis.

    Hybrid Heading and Title Detection:

        Rule-Based Heuristics: A comprehensive set of regular expressions and pattern matching rules are applied to identify common heading structures (e.g., "1. Introduction", "Chapter X", all-caps phrases, specific keywords like "Summary", "Appendix"). These rules are weighted based on their typical significance.

        Machine Learning Classification:

            A TfidfVectorizer transforms text content into numerical feature vectors, capturing the importance of words within the document.

            A lightweight SVC (Support Vector Classifier) is trained to predict whether a given line is a heading or not, based on its textual features.

            For lines identified as potential headings, a RandomForestClassifier is employed to classify their specific hierarchical level (H1, H2, H3, H4).

            Synthetic Training Data: The ML models are trained using carefully crafted synthetic examples that simulate realistic heading and non-heading patterns found in diverse PDF documents. This strategy enhances the models' ability to generalize and perform well even on complex or unconventional PDF layouts.

        Combined Decision: Rule-based scores and machine learning confidence probabilities are intelligently combined using weighted averages. This hybrid approach leverages the strengths of both methods, leading to a more robust and accurate classification. Adaptive confidence thresholds and post-processing filters are applied to refine the final outline, removing false positives and ensuring a logical document structure.

        Title Extraction: A dedicated module focuses on the first few pages of the PDF. It scores potential titles based on factors like position (top of the page), font size (larger), casing (all-caps, title case), and the presence of relevant keywords. An ML component further refines this by considering if a line is not a typical heading, as titles often have unique characteristics. A fallback to PDF metadata is included if text-based extraction is unsuccessful.

    Output Generation:

        The extracted document title and the hierarchical outline (containing level, text, and page number for each detected heading) are formatted into a JSON file.

        Special handling is included for form-like documents (e.g., "Application Form") to return an empty outline, as per common requirements for such document types.

Libraries Used

The solution exclusively utilizes the following open-source Python libraries, as specified in requirements.txt:

    pdfplumber==0.10.3: For robust and efficient PDF text and layout extraction.

    PyPDF2==3.0.1: Included as a backup PDF processing utility, though pdfplumber is the primary extraction engine.

    scikit-learn==1.3.0: Provides the core machine learning algorithms for text vectorization (TfidfVectorizer), classification (SVC, RandomForestClassifier).

    numpy==1.24.3: Essential for high-performance numerical operations, particularly for handling feature arrays in the ML models.

    joblib==1.3.2: Used for efficient saving and loading of Python objects (e.g., trained ML models), ensuring quick startup if models were to be persisted.

Constraints Adherence

The solution is meticulously designed to operate within all specified hackathon constraints for Challenge 1a:

    CPU Only: All processing, including PDF parsing and machine learning computations, is performed entirely on the CPU. There are no GPU dependencies.

    Model Size le200MB: The scikit-learn models and TfidfVectorizer are inherently lightweight. Their combined size (including TfidfVectorizer data) is optimized for memory efficiency and remains well within the 200MB limit.

    Processing Time le10 seconds for a 50-page PDF: The architecture prioritizes efficiency. PDF parsing is performed once, and feature extraction and ML inference are optimized for speed.

    No Internet Access Allowed: The solution is entirely self-contained. It performs no external API calls, web requests, or any form of internet communication during execution. All necessary data and models are part of the Docker image or provided via mounted volumes.

    Runtime Environment: The solution is built for amd64 architecture and is designed to run efficiently on systems with 8 CPUs and 16 GB RAM.

    Output Format: The final JSON output for each processed PDF strictly conforms to the specified schema, including title and outline (with level, text, page).

How to Build and Run

Follow these step-by-step instructions to set up and execute the Challenge 1a solution using Docker.

Prerequisites:

    Docker installed and running on your system.

1. Create the Project Directory:

Start by creating a dedicated directory for your Challenge 1a solution and navigate into it:

mkdir Challenge_1a_Solution
cd Challenge_1a_Solution

2. Create the Code Files:

Save the content of the Python scripts (pdf_extractor.py, process_pdfs.py), requirements.txt, and Dockerfile into their respective files within the Challenge_1a_Solution/ directory. You can copy the content from previous interactions.

    pdf_extractor.py (The updated version from the previous turn, containing AdvancedPDFOutlineExtractor)

    process_pdfs.py (The main script for Challenge 1a processing)

    requirements.txt

    Dockerfile

    README.md (this file)

For process_pdfs.py for Challenge 1a, use the following content:

import os
import json
from pathlib import Path
from pdf_extractor import AdvancedPDFOutlineExtractor # Import the extractor

def process_pdfs():
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the PDF extractor
    extractor = AdvancedPDFOutlineExtractor()
    
    # Train ML models on startup (optional, but recommended for better performance)
    if extractor.ML_AVAILABLE:
        extractor.train_on_examples()

    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        # Extract outline using the AdvancedPDFOutlineExtractor
        result = extractor.extract_outline(pdf_file)
        
        # Remove 'all_lines' if it exists, as it's not part of Challenge 1a output schema
        if 'all_lines' in result:
            del result['all_lines']

        # Create output JSON file
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Found {len(result.get('outline', []))} headings")

if __name__ == "__main__":
    print("Starting Challenge 1a processing")
    process_pdfs() 
    print("Completed Challenge 1a processing")


For requirements.txt for Challenge 1a, use the following content:

pdfplumber==0.10.3
PyPDF2==3.0.1
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.2

For Dockerfile for Challenge 1a, use the following content:

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
COPY process_pdfs.py .

# Create input and output directories as expected by the script
# These directories will be mounted as volumes when running the container.
RUN mkdir -p /app/input /app/output

# Command to run the main processing script for Challenge 1a
CMD ["python", "process_pdfs.py"]

3. Prepare Input and Output Directories:

Create input and output subdirectories inside your Challenge_1a_Solution/ folder.

    input/: This directory must contain your PDF documents for analysis.

        Place your PDF files here (e.g., file01.pdf, file02.pdf, etc.). You can use the sample PDFs from the hackathon's GitHub repository (e.g., Challenge_1a/sample_dataset/pdfs/).

    output/: This directory will be used by the Docker container to write the generated JSON files.

mkdir input
mkdir output

Example of input/ directory content:

Challenge_1a_Solution/
├── input/
│   ├── file01.pdf
│   ├── file02.pdf
│   └── file03.pdf

4. Build the Docker Image:

Navigate to the Challenge_1a_Solution/ directory (where your Dockerfile is located) and execute the Docker build command. Replace <your_repo_name> and <some_identifier> with your chosen image name and tag (e.g., adobe-hackathon-1a-solution:latest).

docker build --platform linux/amd64 -t <your_repo_name>:<some_identifier> .

Example:

docker build --platform linux/amd64 -t adobe-hackathon-1a-solution:latest .

5. Run the Docker Container:

Once the Docker image is successfully built, run the container. This command will mount your local input and output directories as volumes inside the container, enabling the script to read input PDFs and write the output JSON. The --network none flag ensures offline execution.

docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none <your_repo_name>:<some_identifier>

Example:

docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none adobe-hackathon-1a-solution:latest

After the container finishes execution, you will find the generated JSON files (e.g., file01.json, file02.json) in your local Challenge_1a_Solution/output/ directory, each containing the extracted title and outline for the corresponding PDF.
