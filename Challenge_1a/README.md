# PDF Outline Extractor

This solution is designed for the Adobe India Hackathon's "Connecting the Dots" challenge, specifically focusing on Round 1A: "Understand Your Document." The goal is to extract a structured outline (title, H1, H2, and H3 headings with their respective page numbers) from PDF documents and output them in a clean, hierarchical JSON format.

## Approach

The `AdvancedPDFOutlineExtractor` class implements a hybrid approach combining rule-based heuristics with machine learning (ML) enhancements for robust title and heading detection:

1.  **Text and Feature Extraction**: The tool extracts text from each page, identifying properties such as font size, position (`y0`, `x0`), font variation, dominant font, and word count for each line. This detailed feature set is crucial for both rule-based and ML-driven analysis.

2.  **Title Extraction**:
    * It prioritizes the first page of the PDF.
    * It scores potential titles based on factors like position (earlier lines are more likely), text length, casing (all-caps, title case), font size, and the presence of domain-specific keywords.
    * An ML component is integrated to enhance title detection, by giving a higher score to titles that are *not* typical headings.
    * A fallback mechanism extracts titles from PDF metadata if text-based extraction is unsuccessful.
    * Special handling is included for form-like documents to return an empty outline if the title indicates it's an application form.

3.  **Heading Detection (Hybrid - Rules + ML)**:
    * **Rule-Based Scoring**: A set of regular expressions and pattern matching is used to identify common heading structures (e.g., "1. Introduction", "Chapter X", all-caps phrases). Scores are boosted based on font size and vertical position on the page (higher is better).
    * **ML Confidence**: If ML libraries are available, a lightweight `TfidfVectorizer` and `SVC` classifier are used to predict if a line is a heading. A `RandomForestClassifier` then attempts to classify the heading level (H1, H2, H3, H4).
    * **Combined Decision**: The rule-based score and ML confidence score are combined using weighted averages to determine if a line is a heading and its likely level. A decision threshold is applied to filter out low-confidence detections.
    * **Filtering**: Lines that are too short or too long, or match common non-heading patterns (page numbers, dates, copyrights, URLs), are skipped.
    * **Post-processing**: After initial extraction, if a high overall confidence of headings is observed, very low-confidence outliers are removed to refine the outline.

4.  **Output Generation**: The extracted title and hierarchical outline (containing `level`, `text`, and `page` number) are saved as a JSON file in the `/app/output` directory, corresponding to each processed PDF.

## Libraries Used

The solution relies on the following Python libraries, as specified in `requirements.txt`:

* `pdfplumber==0.10.3`: For robust and efficient PDF text and layout extraction.
* `PyPDF2==3.0.1`: As a backup PDF processing library.
* `scikit-learn==1.3.0`: Provides machine learning algorithms for text vectorization (TF-IDF), classification (SVM, RandomForest), and related utilities.
* `numpy==1.24.3`: Essential for numerical operations, especially with feature arrays for ML.
* `joblib==1.3.2`: Used for efficient saving and loading of Python objects, potentially for trained ML models.

## How to Build and Run

The solution is designed to run within a Docker container, adhering to the hackathon's requirements for a self-contained environment, AMD64 architecture, and offline execution without GPU dependencies.

### Build the Docker Image

Navigate to the root directory containing the `Dockerfile` and `requirements.txt` and run the following command to build the Docker image:

```bash
docker build --platform linux/amd64 -t pdf-extractor:latest .