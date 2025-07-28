import os
import json
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pdfplumber # Ensure pdfplumber is imported here for general PDF operations

# Import the AdvancedPDFOutlineExtractor from the new pdf_extractor.py
from pdf_extractor import AdvancedPDFOutlineExtractor

# Lightweight ML dependencies for relevance scoring
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE_RELEVANCE = True
except ImportError:
    ML_AVAILABLE_RELEVANCE = False
    print("Scikit-learn not available for relevance scoring. Relevance will be 0.")

class Challenge1bProcessor:
    def __init__(self):
        self.outline_extractor = AdvancedPDFOutlineExtractor()
        if ML_AVAILABLE_RELEVANCE:
            self.vectorizer_relevance = TfidfVectorizer(stop_words='english', max_features=5000)
            self.is_vectorizer_trained = False
        else:
            self.vectorizer_relevance = None
            self.is_vectorizer_trained = False

    def train_relevance_vectorizer(self, corpus):
        """Train TF-IDF vectorizer on a corpus of text."""
        if ML_AVAILABLE_RELEVANCE and self.vectorizer_relevance and corpus:
            self.vectorizer_relevance.fit(corpus)
            self.is_vectorizer_trained = True
        elif not ML_AVAILABLE_RELEVANCE:
            print("ML_AVAILABLE_RELEVANCE is False, cannot train vectorizer.")
        else:
            print("Warning: No corpus provided to train relevance vectorizer or vectorizer not initialized.")


    def score_relevance(self, text, persona_job_query):
        """Score the relevance of text to the persona/job query."""
        if not ML_AVAILABLE_RELEVANCE or not self.is_vectorizer_trained or not self.vectorizer_relevance:
            return 0.0 # Return 0 if ML not available or not trained

        # Ensure vectorizer has seen the query terms
        # This part is crucial: if query terms are new, the vectorizer needs to be refitted
        # However, refitting within score_relevance is problematic if called multiple times
        # A better approach is to fit it once on all possible text + query.
        # For this challenge, we fit on all document text + query at the start of process_challenge_1b
        
        try:
            text_vec = self.vectorizer_relevance.transform([text])
            query_vec = self.vectorizer_relevance.transform([persona_job_query])

            if text_vec.nnz > 0 and query_vec.nnz > 0: # Check if vectors are not empty
                return cosine_similarity(text_vec, query_vec)[0][0]
            return 0.0
        except Exception as e:
            print(f"Error during relevance scoring: {e}")
            return 0.0


    def extract_and_rank_sections(self, pdf_file_path, persona_job_query):
        """Extract outline and rank sections based on relevance."""
        outline_results = self.outline_extractor.extract_outline(pdf_file_path)
        sections_with_levels = [] # Store sections with their levels for refined text processing

        for item in outline_results.get('outline', []):
            sections_with_levels.append({
                "document": pdf_file_path.name,
                "section_title": item['text'],
                "page_number": item['page'],
                "level": item['level'], 
                "relevance_score": self.score_relevance(item['text'], persona_job_query)
            })
        
        sections_with_levels.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Prepare the list for output, which does not include 'level' or 'relevance_score'
        output_sections = []
        for i, section in enumerate(sections_with_levels):
            output_section = {
                "document": section['document'],
                "section_title": section['section_title'],
                "page_number": section['page_number'],
                "importance_rank": i + 1
            }
            output_sections.append(output_section)
        
        return outline_results.get('title', ''), output_sections, sections_with_levels, outline_results.get('all_lines', [])

    def extract_refined_text(self, document_name, all_document_lines, target_section_title, target_page_number, target_section_level, persona_job_query):
        """
        Extracts relevant paragraphs for a specific section from the pre-extracted lines.
        It finds the content block under the target_section_title until the next heading of same or higher level.
        """
        refined_texts = []
        
        start_index = -1
        # Find the exact start of the target section in all_document_lines
        for i, line_prop in enumerate(all_document_lines):
            # Use a more robust match for section title, considering potential minor text differences
            if line_prop['page_num'] == target_page_number and \
               (line_prop['text'] == target_section_title or \
                re.match(re.escape(target_section_title), line_prop['text'], re.IGNORECASE)):
                start_index = i
                break
        
        if start_index == -1:
            return refined_texts

        level_order = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4}
        target_level_rank = level_order.get(target_section_level, 5)

        end_index = len(all_document_lines)
        # Find the end of the section (next heading of same or higher level)
        for i in range(start_index + 1, len(all_document_lines)):
            line_prop = all_document_lines[i]
            
            is_new_heading_candidate = False
            # Check if this line's text matches any of the heading patterns for levels equal or higher than target section
            for level_key, config in self.outline_extractor.heading_patterns.items(): 
                current_line_level_rank = level_order.get(level_key, 5)
                # A new heading of the same or higher level marks the end of the current section
                if current_line_level_rank <= target_level_rank:
                    for pattern in config['patterns']:
                        if re.match(pattern, line_prop['text'], re.IGNORECASE):
                            # Ensure it's not the same line as the target section title itself (edge case)
                            if not (line_prop['text'] == target_section_title and line_prop['page_num'] == target_page_number):
                                is_new_heading_candidate = True
                                break
                if is_new_heading_candidate:
                    break
            
            if is_new_heading_candidate:
                end_index = i
                break
        
        content_lines_in_section = all_document_lines[start_index + 1 : end_index]

        current_paragraph_lines = []
        current_paragraph_page = None # To track the page of the current paragraph
        
        for line_prop in content_lines_in_section:
            text = line_prop['text'].strip()
            
            # Heuristic for paragraph breaks:
            # 1. Empty line
            # 2. Significant change in x0 (indentation)
            # 3. Sudden large change in y0 (vertical spacing) - implies new block
            # 4. Line is very short and doesn't end with punctuation, might be a list item or caption
            
            is_paragraph_break = False
            if not text: # Empty line
                is_paragraph_break = True
            elif current_paragraph_lines and line_prop['x0'] > (content_lines_in_section[content_lines_in_section.index(line_prop)-1]['x0'] + 10): # Significant indent
                is_paragraph_break = True
            elif current_paragraph_lines and line_prop['y0'] < (content_lines_in_section[content_lines_in_section.index(line_prop)-1]['y0'] - 20): # Large vertical jump
                 is_paragraph_break = True
            # Add more heuristics for complex PDFs: e.g., if line is very short and doesn't end with common punctuation
            elif len(text.split()) < 5 and not re.search(r'[.!?]$', text) and not text.endswith(':'):
                 pass # Don't break if it's a short line that might be part of a sentence
            
            if is_paragraph_break and current_paragraph_lines:
                paragraph_content = " ".join(current_paragraph_lines)
                # Only add meaningful paragraphs and those relevant to the query
                if len(paragraph_content.split()) > 10 and self.score_relevance(paragraph_content, persona_job_query) > 0.05: # Lower threshold for content
                    refined_texts.append({'text': paragraph_content, 'page': current_paragraph_page})
                current_paragraph_lines = []
                current_paragraph_page = None
            
            if text: # Add non-empty lines to current paragraph
                current_paragraph_lines.append(text)
                if current_paragraph_page is None: # Set page for the start of a new paragraph
                    current_paragraph_page = line_prop['page_num']

        # Add the last accumulated paragraph if any
        if current_paragraph_lines: 
            paragraph_content = " ".join(current_paragraph_lines)
            if len(paragraph_content.split()) > 10 and self.score_relevance(paragraph_content, persona_job_query) > 0.05:
                refined_texts.append({'text': paragraph_content, 'page': current_paragraph_page})

        scored_and_paged_paragraphs = []
        for para_info in refined_texts:
            score = self.score_relevance(para_info['text'], persona_job_query)
            scored_and_paged_paragraphs.append((score, para_info['text'], para_info['page']))
        
        scored_and_paged_paragraphs.sort(key=lambda x: x[0], reverse=True)

        final_refined_texts_for_output = []
        char_count = 0
        MAX_CHARS_PER_SUBSECTION = 1000 # Increased limit for more content
        MAX_SUBSECTION_ENTRIES = 5 # More entries per main section

        for score, text_content, page_num in scored_and_paged_paragraphs:
            if len(final_refined_texts_for_output) >= MAX_SUBSECTION_ENTRIES:
                break
            
            # Ensure snippet doesn't exceed total max chars, and is not too short
            if char_count + len(text_content) <= MAX_CHARS_PER_SUBSECTION and len(text_content) > 50: # Minimum length for a snippet
                final_refined_texts_for_output.append({
                    "document": document_name,
                    "refined_text": text_content,
                    "page_number": page_num
                })
                char_count += len(text_content)
            elif char_count < MAX_CHARS_PER_SUBSECTION and len(text_content) > 50: # Can still fit a partial text
                remaining_chars = MAX_CHARS_PER_SUBSECTION - char_count
                if remaining_chars > 100: # Ensure snippet is meaningful
                    snippet = text_content[:remaining_chars] + "..."
                    final_refined_texts_for_output.append({
                        "document": document_name,
                        "refined_text": snippet,
                        "page_number": page_num
                    })
                    char_count += len(snippet)

        return final_refined_texts_for_output


    def process_challenge_1b(self, input_dir="/app/input", output_dir="/app/output"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        challenge_input_file = input_path / "challenge1b_input.json"
        if not challenge_input_file.exists():
            print(f"Error: challenge1b_input.json not found in {input_dir}")
            return

        with open(challenge_input_file, 'r', encoding='utf-8') as f:
            challenge_data = json.load(f)

        persona = challenge_data.get("persona", {}).get("role", "")
        job_to_be_done = challenge_data.get("job_to_be_done", {}).get("task", "")
        input_documents_meta = challenge_data.get("documents", [])
        
        # Extract just filenames for input_documents list in metadata
        input_document_names = [doc['filename'] for doc in input_documents_meta]

        if not input_documents_meta:
            print("No input documents specified in challenge1b_input.json")
            return

        # Train ML models for outline extraction once
        if self.outline_extractor.ML_AVAILABLE:
            self.outline_extractor.train_on_examples()

        # Prepare corpus for relevance vectorizer training
        corpus_for_relevance = [f"{persona} {job_to_be_done}"] # Start with the query itself
        all_document_lines_map = {} # Store all lines for each document for later refined text extraction

        for doc_meta in input_documents_meta:
            doc_name = doc_meta['filename']
            doc_path = input_path / doc_name
            if doc_path.exists():
                try:
                    with pdfplumber.open(doc_path) as pdf:
                        lines_from_pdf = []
                        for page_num, page in enumerate(pdf.pages, 1):
                            if page_num > 50: # Limit pages to process per PDF
                                break
                            text_lines_on_page = self.outline_extractor._extract_text_with_features(page)
                            for line_prop in text_lines_on_page:
                                line_prop['page_num'] = page_num # Add page number to line properties
                                lines_from_pdf.append(line_prop)
                                corpus_for_relevance.append(line_prop['text']) # Add text to corpus
                        all_document_lines_map[doc_name] = lines_from_pdf
                except Exception as e:
                    print(f"Could not read {doc_name} for corpus or outline: {e}")
        
        self.train_relevance_vectorizer(corpus_for_relevance)

        all_extracted_sections_output = []
        all_subsection_analysis_output = []

        for doc_meta in input_documents_meta:
            doc_name = doc_meta['filename']
            pdf_file_path = input_path / doc_name
            
            if not pdf_file_path.exists() or doc_name not in all_document_lines_map:
                print(f"Warning: Document {doc_name} not found or lines not extracted, skipping.")
                continue

            # Pass the already extracted all_lines for the document
            doc_title, sections_for_output, sections_with_levels, _ = \
                self.extract_and_rank_sections(pdf_file_path, f"{persona} {job_to_be_done}")
            
            # Only add top N sections to all_extracted_sections_output as per sample output (e.g., top 5)
            # Ensure sections_for_output is not empty before extending
            if sections_for_output:
                all_extracted_sections_output.extend(sections_for_output[:5]) 

            # For each top section, perform sub-section analysis
            if sections_with_levels: # Ensure there are sections to process
                for section_info in sections_with_levels[:5]: # Use top 5 sections for sub-section analysis
                    refined_texts = self.extract_refined_text(
                        doc_name, # Pass document name
                        all_document_lines_map[doc_name], # Pass all lines for the current document
                        section_info['section_title'], 
                        section_info['page_number'], 
                        section_info['level'], 
                        f"{persona} {job_to_be_done}"
                    )
                    all_subsection_analysis_output.extend(refined_texts)
        
        # Sort all extracted sections by importance_rank across all documents
        # This will re-rank sections globally if multiple documents are processed
        all_extracted_sections_output.sort(key=lambda x: x['importance_rank'])

        # Final output structure
        output_data = {
            "metadata": {
                "input_documents": input_document_names, # Use just names for metadata
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": all_extracted_sections_output,
            "subsection_analysis": all_subsection_analysis_output
        }

        # The output file name is fixed for challenge 1b as challenge1b_output.json
        output_file_name = "challenge1b_output.json" 
        output_file_path = output_path / output_file_name
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"Challenge 1b output saved to: {output_file_path.name}")

def main():
    processor = Challenge1bProcessor()
    processor.process_challenge_1b()


if __name__ == "__main__":
    print("Starting Challenge 1b processing")
    main()
    print("Completed Challenge 1b processing")

