Approach Explanation for Challenge 1b: Persona-Driven Document Intelligence
Methodology Overview

The "Persona-Driven Document Intelligence" solution is engineered to transform a collection of raw PDF documents into actionable insights, tailored to a specific user persona and their defined "job-to-be-done." Our methodology integrates robust document understanding with advanced natural language processing techniques to identify, rank, and extract the most relevant information, even from complex and varied document sets.

At its core, the system operates in three main phases: comprehensive PDF content extraction, persona-driven relevance scoring and global section ranking, and granular sub-section analysis for refined text extraction.
1. Comprehensive PDF Content Extraction

This foundational phase leverages a sophisticated document outline extraction module. For each input PDF, pdfplumber is employed to perform a deep parse, capturing not only the raw text but also critical layout features for every line. These features include font size, precise spatial coordinates (x and y positions), font variation (indicating consistency), and word count. This rich metadata is crucial for accurately discerning document structure beyond simple text content.

To identify titles and hierarchical headings (H1, H2, H3, H4), a hybrid approach is utilized. This combines a meticulously crafted set of rule-based heuristics (e.g., pattern matching for numbered headings, all-caps phrases, and specific keywords) with lightweight machine learning models from scikit-learn. A TfidfVectorizer transforms text into numerical representations, while an SVC classifier determines if a line is a heading, and a RandomForestClassifier predicts its hierarchical level. Synthetic, yet realistic, training data is generated to ensure these models generalize effectively across diverse PDF layouts and content types. The final decision for heading identification and leveling intelligently blends rule-based scores with ML confidence probabilities, followed by post-processing to ensure a coherent and logical document outline.
2. Persona-Driven Relevance Scoring and Global Section Ranking

Once the structural outline of each document is established, the system shifts its focus to content relevance. The persona (role) and job-to-be-done (task) provided in the challenge1b_input.json serve as the central query. All textual content from the input documents, along with this query, is used to train a TfidfVectorizer. This ensures that the vocabulary and term weighting are highly specific and contextual to the given document collection and user's intent.

The semantic relevance between each extracted section title and the combined persona/job query is then calculated using cosine similarity. This metric quantifies how closely the meaning of a section aligns with the user's stated needs. All identified sections from across all input documents are then globally ranked based on these relevance scores. The top-ranked sections are selected to form the primary "extracted sections" output, each assigned an importance_rank.
3. Granular Sub-section Analysis for Refined Text

For the most relevant sections identified in the previous phase, a deeper dive is performed to extract precise content snippets. The system intelligently navigates the raw lines of text within each section's boundaries (from its heading to the next heading of equal or higher hierarchy). Heuristics are applied to segment this content into logical paragraphs (e.g., detecting empty lines, significant indentation changes, or large vertical spacing shifts).

Each of these candidate paragraphs is then re-scored for relevance against the persona and job-to-be-done. The most pertinent paragraphs, up to a defined character and entry limit, are selected as "refined text." This ensures that the subsection_analysis output provides concise, highly relevant, and actionable information, directly addressing the user's query without unnecessary verbosity.
Adherence to Constraints

The solution is meticulously designed to operate within all specified hackathon constraints. It is built for linux/amd64 CPU architecture, with no GPU dependencies. All scikit-learn models are lightweight, ensuring the total model size remains well under the 1GB limit. The processing pipeline is optimized for speed, aiming to meet the 60-second time limit for document collections by avoiding redundant processing and utilizing efficient algorithms. Crucially, the entire solution operates offline, making no external network calls during execution, ensuring a self-contained and reliable system.
