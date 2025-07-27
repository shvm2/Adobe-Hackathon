# PDF Outline Extractor

## Approach

This solution extracts structured outlines from PDF documents using the following approach:

1. **Title Extraction**: Analyzes the first page to identify the document title
2. **Heading Detection**: Uses pattern matching and text analysis to identify headings:
   - H1: Chapter titles, numbered sections (1. Introduction)
   - H2: Subsections (1.1 Overview)
   - H3: Sub-subsections (1.1.1 Details)

3. **Pattern Recognition**: Combines multiple strategies:
   - Regex patterns for numbered headings
   - Case analysis (uppercase, title case)
   - Text length constraints

## Libraries Used

- `pdfplumber`: For robust PDF text extraction
- `PyPDF2`: Backup PDF processing

## How to Build and Run

Build the Docker image:
```bash
docker build --platform linux/amd64 -t pdf-extractor:latest .