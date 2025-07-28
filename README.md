Adobe India Hackathon 2025 - Connecting the Dots Challenge
Welcome to the "Connecting the Dots" Challenge

Rethink Reading. Rediscover Knowledge.

In a world flooded with documents, what wins is not more content — it's context. This hackathon challenges us to reimagine the humble PDF as an intelligent, interactive experience—one that understands structure, surfaces insights, and responds like a trusted research companion. This repository contains the solutions developed for Round 1 of this challenge, focusing on document understanding and persona-driven intelligence.
The Journey Ahead (Round 1)

Round 1: Building the Brains
This round focuses on extracting structured outlines from raw PDFs with blazing speed and pinpoint accuracy, and then powering it up with on-device intelligence that understands sections and links related ideas together based on specific user needs.
Challenge Solutions

This repository is structured to hold the solutions for both sub-challenges of Round 1:
1. Challenge 1a: Understand Your Document

Objective: To extract a structured outline (Title, H1, H2, H3, H4 headings with page numbers) from individual PDF documents in a clean, hierarchical JSON format. This forms the fundamental document understanding layer.

Key Features:

    Hybrid Heading Detection: Combines rule-based heuristics with machine learning (TF-IDF, SVC, RandomForestClassifier) for high accuracy.

    Detailed Feature Extraction: Utilizes font size, position, casing, and other layout features for robust classification.

    Optimized Performance: Designed to meet strict time and memory constraints for processing single PDFs.

    Offline Execution: Fully self-contained, requiring no internet access during runtime.

For detailed information on Challenge 1a, including its approach, libraries, constraints, and how to build/run, please refer to the dedicated README.md in the Challenge_1a_Solution directory.
2. Challenge 1b: Persona-Driven Document Intelligence

Objective: To build an intelligent document analyst that extracts and prioritizes the most relevant sections and sub-sections from a collection of documents based on a specific user persona and their job-to-be-done.

Key Features:

    Multi-Document Processing: Handles a collection of related PDFs.

    Persona-Driven Relevance: Uses TF-IDF and Cosine Similarity to score content relevance against user persona and job-to-be-done.

    Section Ranking: Ranks extracted sections by importance.

    Refined Sub-section Analysis: Extracts and prioritizes granular content snippets within relevant sections.

    Adherence to Constraints: Meets the specified performance, model size, and offline execution requirements for multi-document analysis.

For detailed information on Challenge 1b, including its approach, libraries, constraints, and how to build/run, please refer to the dedicated README.md in the Challenge_1b_Solution directory.
Repository Structure

.
├── Challenge_1a_Solution/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   ├── pdf_extractor.py
│   └── process_pdfs.py
├── Challenge_1b_Solution/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   ├── pdf_extractor.py
│   └── process_challenge_1b.py
├── README.md              <-- This file
└── .gitignore             # (Optional) Standard Git ignore file

Overall Constraints (Applicable to Both Challenges)

    CPU Architecture: All solutions must be compatible with linux/amd64 (x86_64).

    No GPU Dependencies: Solutions must run purely on CPU.

    Offline Execution: No internet access is allowed during the execution of the Docker containers.

    Resource Limits: Solutions are designed to run on systems with 8 CPUs and 16 GB RAM.
