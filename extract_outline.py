import os
import json
import pdfplumber
from pathlib import Path
import re
from collections import Counter


class PDFOutlineExtractor:
    def __init__(self):
        # Simplified and more flexible heading patterns
        self.heading_patterns = [
            # Numbered patterns
            r'^\s*\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\s*\d+\.\d+\s+[A-Z]',        # "1.1 Subsection" 
            r'^\s*\d+\.\d+\.\d+\s+[A-Z]',  # "1.1.1 Sub-subsection"
            r'^\s*Chapter\s+\d+\s*:?\s*[A-Z]', # "Chapter 1: Title"
            r'^\s*CHAPTER\s+\d+\s*:?\s*[A-Z]', # "CHAPTER 1: TITLE"
            
            # All caps patterns (likely H1)
            r'^[A-Z][A-Z\s\d\-\.\,\:\;]+$',    # "INTRODUCTION", "PROBLEM STATEMENT"
            
            # Title case patterns
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*\s*$', # "Introduction", "Problem Statement"
            
            # Mixed patterns that look like headings
            r'^[A-Z][a-zA-Z\s\-\.\,\:\;]*[a-zA-Z]$' # General heading pattern
        ]
        
        # Minimum confidence for heading detection
        self.min_heading_length = 3
        self.max_heading_length = 100

    def _extract_text_multiple_methods(self, page):
        """Try multiple methods to extract text from a page."""
        text_lines = []
        
        # Method 1: Try detailed word extraction
        try:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if words:
                # Group words into lines based on y-coordinate
                lines = {}
                for word in words:
                    if 'y0' not in word or 'text' not in word:
                        continue
                        
                    # Use more precise rounding for grouping
                    y_key = round(word['y0'], 1)
                    if y_key not in lines:
                        lines[y_key] = []
                    lines[y_key].append(word)
                
                # Convert to list of line objects
                for y_pos in sorted(lines.keys(), reverse=True):  # Top to bottom
                    line_words = sorted(lines[y_pos], key=lambda w: w.get('x0', 0))
                    
                    text = ' '.join([w['text'] for w in line_words if w.get('text', '').strip()])
                    if not text.strip():
                        continue
                    
                    # Calculate average font size
                    sizes = [w.get('size', 12) for w in line_words if w.get('size') and w.get('size') > 0]
                    avg_size = sum(sizes) / len(sizes) if sizes else 12
                    
                    text_lines.append({
                        'text': text.strip(),
                        'size': avg_size,
                        'y0': y_pos
                    })
                
                if text_lines:
                    print(f"Method 1 (detailed words): Extracted {len(text_lines)} lines")
                    return text_lines
                    
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Simple text extraction with line splitting
        try:
            text = page.extract_text()
            if text and text.strip():
                lines = text.split('\n')
                text_lines = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        text_lines.append({
                            'text': line,
                            'size': 12,  # Default size
                            'y0': len(lines) - i  # Approximate y position
                        })
                
                if text_lines:
                    print(f"Method 2 (simple text): Extracted {len(text_lines)} lines")
                    return text_lines
                    
        except Exception as e:
            print(f"Method 2 failed: {e}")
        
        # Method 3: Character-level extraction
        try:
            chars = page.chars
            if chars:
                # Group characters into lines
                lines = {}
                for char in chars:
                    if 'y0' not in char or 'text' not in char:
                        continue
                    
                    y_key = round(char['y0'], 1)
                    if y_key not in lines:
                        lines[y_key] = []
                    lines[y_key].append(char)
                
                text_lines = []
                for y_pos in sorted(lines.keys(), reverse=True):
                    line_chars = sorted(lines[y_pos], key=lambda c: c.get('x0', 0))
                    text = ''.join([c['text'] for c in line_chars if c.get('text', '')])
                    
                    if text.strip():
                        sizes = [c.get('size', 12) for c in line_chars if c.get('size') and c.get('size') > 0]
                        avg_size = sum(sizes) / len(sizes) if sizes else 12
                        
                        text_lines.append({
                            'text': text.strip(),
                            'size': avg_size,
                            'y0': y_pos
                        })
                
                if text_lines:
                    print(f"Method 3 (characters): Extracted {len(text_lines)} lines")
                    return text_lines
                    
        except Exception as e:
            print(f"Method 3 failed: {e}")
        
        print("All text extraction methods failed")
        return []

    def extract_title(self, pdf):
        """Extract document title from first page with multiple fallback strategies."""
        print("\n=== Title Extraction Debug ===")
        
        try:
            first_page = pdf.pages[0]
            text_lines = self._extract_text_multiple_methods(first_page)
            
            print(f"Extracted {len(text_lines)} text lines")
            
            if not text_lines:
                print("No text lines found, falling back to metadata")
                return self._get_metadata_title(pdf)
            
            # Debug: Print first few lines
            print("First 5 lines extracted:")
            for i, line in enumerate(text_lines[:5]):
                print(f"  Line {i+1}: '{line['text']}' (size: {line['size']})")
            
            # Strategy 1: Look for title in first few lines with relaxed criteria
            print("\nStrategy 1: Looking for title characteristics...")
            for i, line in enumerate(text_lines[:10]):
                text = line['text'].strip()
                
                # Skip very short or very long text
                if len(text) < 5 or len(text) > 150:
                    print(f"  Skipping line {i+1}: length {len(text)} out of range")
                    continue
                
                # Skip obvious non-titles
                if re.match(r'^\s*(page\s+\d+|abstract|introduction|chapter\s+\d+)\s*$', text, re.IGNORECASE):
                    print(f"  Skipping line {i+1}: matches non-title pattern")
                    continue
                
                # More relaxed title detection
                has_alpha = bool(re.search(r'[a-zA-Z]', text))
                starts_with_cap = text and text[0].isupper()
                reasonable_length = 10 <= len(text) <= 100
                
                print(f"  Line {i+1} analysis:")
                print(f"    Text: '{text}'")
                print(f"    Has alpha: {has_alpha}")
                print(f"    Starts with cap: {starts_with_cap}")
                print(f"    Reasonable length: {reasonable_length}")
                
                if has_alpha and starts_with_cap and reasonable_length:
                    print(f"  -> Selected as title: '{text}'")
                    return text
            
            # Strategy 2: Just use first substantial line
            print("\nStrategy 2: Using first substantial line...")
            for i, line in enumerate(text_lines[:5]):
                text = line['text'].strip()
                if len(text) >= 5 and len(text) <= 150:
                    print(f"  -> Using line {i+1} as title: '{text}'")
                    return text
            
            # Strategy 3: Fallback to metadata
            print("\nStrategy 3: Falling back to metadata...")
            return self._get_metadata_title(pdf)
            
        except Exception as e:
            print(f"Error in title extraction: {e}")
            return self._get_metadata_title(pdf)

    def _get_metadata_title(self, pdf):
        """Get title from PDF metadata, with filtering."""
        try:
            if (pdf.metadata and 'Title' in pdf.metadata and 
                pdf.metadata['Title'] and 
                str(pdf.metadata['Title']).strip()):
                
                title = str(pdf.metadata['Title']).strip()
                
                # Filter out obvious filename patterns
                if not re.match(r'.+\.(doc|docx|pdf)$', title, re.IGNORECASE):
                    print(f"Using metadata title: '{title}'")
                    return title
                else:
                    print(f"Skipping filename-like metadata: '{title}'")
        except Exception as e:
            print(f"Error reading metadata: {e}")
        
        print("Using default title")
        return "Untitled Document"

    def _is_heading(self, text, font_size):
        """Determine if a line of text is likely a heading."""
        text = text.strip()
        
        # Basic filters
        if (len(text) < self.min_heading_length or 
            len(text) > self.max_heading_length or
            not re.search(r'[a-zA-Z]', text)):
            return False
        
        # Skip page numbers and other obvious non-headings
        if re.match(r'^\s*\d+\s*$', text):
            return False
        
        # Check against heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Additional heuristics
        word_count = len(text.split())
        
        # Very likely headings
        if (text.isupper() and word_count <= 8) or \
           (text.istitle() and word_count <= 6) or \
           (font_size and font_size > 14):
            return True
        
        # Numbered sections
        if re.match(r'^\s*\d+[\.\)]\s+[A-Z]', text):
            return True
        
        return False

    def _determine_heading_level(self, text, font_size):
        """Determine the heading level (H1, H2, H3)."""
        text = text.strip()
        
        # H1 patterns (main sections)
        h1_patterns = [
            r'^\s*\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\s*Chapter\s+\d+',           # "Chapter 1"
            r'^\s*CHAPTER\s+\d+',           # "CHAPTER 1"
            r'^[A-Z][A-Z\s\d\-\.\,\:\;]{10,}$'  # Long all-caps
        ]
        
        # H2 patterns (subsections)
        h2_patterns = [
            r'^\s*\d+\.\d+\s+[A-Z]',        # "1.1 Subsection"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*){1,4}\s*$'  # "Problem Statement"
        ]
        
        # H3 patterns (sub-subsections)
        h3_patterns = [
            r'^\s*\d+\.\d+\.\d+\s+[A-Z]',  # "1.1.1 Sub-subsection"
        ]
        
        # Check patterns in order
        for pattern in h1_patterns:
            if re.match(pattern, text):
                return "H1"
        
        for pattern in h2_patterns:
            if re.match(pattern, text):
                return "H2"
        
        for pattern in h3_patterns:
            if re.match(pattern, text):
                return "H3"
        
        # Font size based determination (if available)
        if font_size:
            if font_size >= 16:
                return "H1"
            elif font_size >= 14:
                return "H2"
            elif font_size >= 12:
                return "H3"
        
        # Default heuristics
        if text.isupper():
            return "H1"
        elif text.istitle() and len(text.split()) <= 4:
            return "H2"
        else:
            return "H3"

    def extract_outline(self, pdf_path):
        """Extract structured outline from PDF."""
        print(f"\n=== Processing {pdf_path} ===")
        outline = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                title = self.extract_title(pdf)
                print(f"Final title: '{title}'")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    if page_num > 50:  # Limit as per requirements
                        break
                        
                    text_lines = self._extract_text_multiple_methods(page)
                    
                    for line in text_lines:
                        text = line['text']
                        font_size = line.get('size', 12)
                        
                        if self._is_heading(text, font_size):
                            level = self._determine_heading_level(text, font_size)
                            outline.append({
                                "level": level,
                                "text": text,
                                "page": page_num
                            })
                
                print(f"Found {len(outline)} headings")
                return {
                    "title": title,
                    "outline": outline
                }
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }

    def process_all_pdfs(self, input_dir="/app/input", output_dir="/app/output"):
        """Process all PDFs in input directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            # Extract outline
            result = self.extract_outline(pdf_file)
            
            # Save with correct format (simple format for Round 1A)
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Saved: {output_file.name}")
            print(f"Title: {result['title']}")
            print(f"Outline entries: {len(result['outline'])}")


def main():
    extractor = PDFOutlineExtractor()
    extractor.process_all_pdfs()


if __name__ == "__main__":
    main()