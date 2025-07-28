Adobe India Hackathon 2025 - Challenge 1b Solution: Persona-Driven Document Intelligence
Overview

This project presents a robust and intelligent document analysis system for Challenge 1b: Persona-Driven Document Intelligence. The solution is designed to act as an advanced analyst, capable of processing a collection of PDF documents, extracting relevant sections, and prioritizing them based on a defined user persona and their specific job-to-be-done. It aims to surface critical insights efficiently from complex document sets.

The system builds upon a sophisticated PDF outline extraction module (developed in Challenge 1a) and enhances it with advanced natural language processing (NLP) techniques for content relevance scoring and granular sub-section analysis.
Approach

The solution employs a multi-stage, hybrid approach to achieve accurate and context-aware document analysis:

    Robust PDF Content Extraction (pdf_extractor.py):

        Detailed Text and Feature Extraction: Utilizes pdfplumber to parse PDF documents, extracting not just raw text but also rich layout features for each line (e.g., font size, vertical and horizontal position, font variation, casing, word count). This detailed feature set is crucial for both rule-based and machine learning models.

        Hybrid Heading and Title Detection:

            Rule-Based Heuristics: A comprehensive set of regular expressions and pattern matching identifies common heading structures (e.g., numbered headings, all-caps titles, specific keywords). These rules are assigned weights to reflect their typical importance.

            Machine Learning Classification:

                A TfidfVectorizer transforms text into numerical feature vectors.

                A lightweight SVC (Support Vector Classifier) is trained to classify whether a given line is a heading or not.

                A RandomForestClassifier is then used to predict the hierarchical level (H1, H2, H3, H4) of identified headings.

                Synthetic training data, carefully crafted to represent realistic heading and non-heading patterns across various PDF structures, is used to train these models. This ensures the models generalize well to diverse and complex documents.

            Combined Decision: Rule-based scores and ML confidence probabilities are intelligently combined using weighted averages to make a final, robust decision on whether a line is a heading and its most probable level. Adaptive thresholds and post-processing filters further refine the extracted outline, removing noise and ensuring logical hierarchy.

        Title Extraction: A dedicated title extraction module analyzes the first few pages, scoring potential titles based on position, font size, casing, and domain-specific keywords. An ML component further refines this by favoring text that is descriptive but not necessarily a typical section heading.

    Persona-Driven Relevance and Ranking (process_challenge_1b.py):

        Input Parsing: Reads the challenge1b_input.json to obtain the persona (role) and job_to_be_done (task).

        Contextual Vectorization: A TfidfVectorizer is dynamically trained on the entire corpus of text from all input documents combined with the persona and job-to-be-done query. This ensures the vectorizer's vocabulary and term weights are highly relevant to the specific document collection and user query.

        Relevance Scoring: Cosine similarity is employed to calculate the semantic relevance between the text of each extracted section title and content paragraph against the combined persona and job-to-be-done query. This metric quantifies how well a section's content aligns with the user's needs.

        Global Section Ranking: All extracted sections from all documents are ranked globally based on their relevance scores. The top N most relevant sections are selected and assigned an importance_rank.

    Refined Sub-section Analysis (process_challenge_1b.py):

        For each of the top-ranked main sections, the system performs a more granular extraction of content.

        It identifies the textual content directly under the section's heading, continuing until the next heading of the same or a higher level is encountered.

        This content is then intelligently segmented into paragraphs using heuristics (e.g., empty lines, significant indentation changes, vertical spacing).

        Each identified paragraph is re-scored for relevance against the persona and job-to-be-done query.

        The most relevant paragraphs (up to a predefined character and entry limit) are selected to form the refined_text for the subsection_analysis output. This ensures that only the most pertinent and concise information is presented.

Libraries Used

The solution exclusively utilizes the following open-source Python libraries, as specified in requirements.txt:

    pdfplumber==0.10.3: For robust and efficient PDF text and layout extraction.

    PyPDF2==3.0.1: Included as a backup PDF processing utility, though pdfplumber is the primary extraction engine.

    scikit-learn==1.3.0: Provides the core machine learning algorithms for text vectorization (TfidfVectorizer), classification (SVC, RandomForestClassifier), and similarity calculations (cosine_similarity).

    numpy==1.24.3: Essential for high-performance numerical operations, particularly for handling feature arrays in the ML models.

    joblib==1.3.2: Used for efficient saving and loading of Python objects (e.g., trained ML models), ensuring quick startup if models were to be persisted.

Constraints Adherence

The solution is meticulously designed to operate within all specified hackathon constraints:

    CPU Only: All processing, including PDF parsing and machine learning computations, is performed entirely on the CPU. There are no GPU dependencies.

    Model Size le1GB: The scikit-learn models and TfidfVectorizer are inherently lightweight. The data structures they utilize are optimized for memory efficiency, ensuring the total model footprint remains well within the 1GB limit.

    Processing Time le60 seconds for 3-5 documents: The architecture is optimized for speed. PDF parsing is performed efficiently, and relevance scoring uses fast matrix operations (cosine_similarity). The ML models are kept small (n_estimators=50, max_features=1000) to minimize inference time.

    No Internet Access Allowed: The solution is entirely self-contained. It performs no external API calls, web requests, or any form of internet communication during execution. All necessary data and models are part of the Docker image or provided via mounted volumes.

    Runtime Environment: The solution is built for amd64 architecture and is designed to run efficiently on systems with 8 CPUs and 16 GB RAM.

    Output Format: The final challenge1b_output.json strictly conforms to the specified JSON schema, including metadata, extracted_sections (with importance_rank), and subsection_analysis (with refined_text).

How to Build and Run

Follow these step-by-step instructions to set up and execute the Challenge 1b solution using Docker.

Prerequisites:

    Docker installed and running on your system.

1. Create the Project Directory:

Start by creating a dedicated directory for your Challenge 1b solution and navigate into it:

mkdir Challenge_1b_Solution
cd Challenge_1b_Solution

2. Create the Code Files:

Save the content of the Python scripts (pdf_extractor.py, process_challenge_1b.py), requirements.txt, and Dockerfile into their respective files within the Challenge_1b_Solution/ directory. You can copy the content from the previous responses.

    pdf_extractor.py

    process_challenge_1b.py

    requirements.txt

    Dockerfile

    README.md (this file)

3. Prepare Input and Output Directories:

Create input and output subdirectories inside your Challenge_1b_Solution/ folder.

    input/: This directory must contain your PDF documents for analysis and the challenge1b_input.json file.

        Place all PDF files specified in your challenge1b_input.json here (e.g., South of France - Cities.pdf, Dinner Ideas - Sides_1.pdf).

        Place your challenge1b_input.json file here. You can use the sample challenge1b_input.json files provided in the hackathon's GitHub repository (e.g., Challenge_1b/Collection 1/challenge1b_input.json).

    output/: This directory will be used by the Docker container to write the generated challenge1b_output.json file.

mkdir input
mkdir output

Example of input/ directory content (for "Travel Planner" test case):

Challenge_1b_Solution/
├── input/
│   ├── South of France - Cities.pdf
│   ├── South of France - Cuisine.pdf
│   ├── South of France - History.pdf
│   ├── South of France - Restaurants and Hotels.pdf
│   ├── South of France - Things to Do.pdf
│   ├── South of France - Tips and Tricks.pdf
│   ├── South of France - Traditions and Culture.pdf
│   └── challenge1b_input.json  # Example input for 'Travel Planner' persona

4. Build the Docker Image:

Navigate to the Challenge_1b_Solution/ directory (where your Dockerfile is located) and execute the Docker build command. Replace <your_repo_name> and <some_identifier> with your chosen image name and tag (e.g., adobe-hackathon-1b-solution:latest).

docker build --platform linux/amd64 -t <your_repo_name>:<some_identifier> .

Example:

docker build --platform linux/amd64 -t adobe-hackathon-1b-solution:latest .

5. Run the Docker Container:

Once the Docker image is successfully built, run the container. This command will mount your local input and output directories as volumes inside the container, enabling the script to read input PDFs and write the output JSON. The --network none flag ensures offline execution.

docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none <your_repo_name>:<some_identifier>

Example:

docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none adobe-hackathon-1b-solution:latest

After the container finishes execution, you will find the challenge1b_output.json file in your local Challenge_1b_Solution/output/ directory, containing the persona-driven analysis results.