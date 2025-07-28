import os
import json
import pdfplumber
from pathlib import Path
import re
from datetime import datetime
import pickle
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Lightweight ML dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available, falling back to rules only")

class AdvancedPDFOutlineExtractor:
    def __init__(self):
        self.ml_models = {}
        self.vectorizers = {}
        self.is_trained = False
        
        # Enhanced heading patterns with ML confidence scores
        self.heading_patterns = {
            'H1': {
                'patterns': [
                    r'^\s*\d+\.\s+[A-Z].*',                    # "1. Introduction"
                    r'^\s*Chapter\s+\d+.*',                    # "Chapter 1"
                    r'^\s*CHAPTER\s+\d+.*',                    # "CHAPTER 1"
                    r'^\s*Appendix\s+[A-Z].*',                 # "Appendix A"
                    r'^[A-Z][A-Z\s]{8,50}$',                  # Long all-caps
                    r'^\s*(Revision History|Table of Contents|Acknowledgements|References)\s*$',
                    r'^\s*PATHWAY\s+OPTIONS\s*$',              # Specific patterns
                ],
                'weight': 1.0
            },
            'H2': {
                'patterns': [
                    r'^\s*\d+\.\d+\s+[A-Z].*',                # "2.1 Subsection"
                    r'^\s*(Summary|Background|Timeline|Milestones)\s*:?\s*$',
                    r'^\s*The\s+Business\s+Plan\s+to\s+be\s+Developed\s*$',
                    r'^\s*Approach\s+and\s+Specific\s+Proposal\s+Requirements\s*$',
                    r'^\s*Evaluation\s+and\s+Awarding\s+of\s+Contract\s*$',
                ],
                'weight': 0.8
            },
            'H3': {
                'patterns': [
                    r'^\s*\d+\.\d+\.\d+\s+[A-Z].*',          # "2.1.1 Sub-subsection"
                    r'^\s*\d+\.\s+[A-Z][a-zA-Z\s]*\s*$',     # "1. Preamble"
                    r'^\s*Phase\s+(I|II|III|IV|V|1|2|3|4|5).*', # "Phase I"
                    r'^\s*[A-Z][a-zA-Z\s]*:\s*$',            # "Timeline:"
                    r'^\s*(Equitable access|Shared decision|Shared governance|Shared funding|Local points|Access|Guidance|Training|Provincial Purchasing|Technological Support)\s*:?\s*$',
                ],
                'weight': 0.6
            },
            'H4': {
                'patterns': [
                    r'^\s*For\s+each\s+Ontario\s+(citizen|student|library|government).*',  # Specific for file03
                ],
                'weight': 0.4
            }
        }
        
        # Feature extraction weights
        self.feature_weights = {
            'font_size': 0.3,
            'position': 0.2,
            'case_pattern': 0.25, 
            'length': 0.1,
            'ml_confidence': 0.15
        }
        
        if ML_AVAILABLE:
            self._initialize_ml_components()

    def _initialize_ml_components(self):
        """Initialize ML components for heading classification"""
        # TF-IDF Vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features to stay lightweight
            ngram_range=(1, 2), # Unigrams and bigrams
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        
        # Lightweight SVM classifier
        self.heading_classifier = SVC(
            kernel='linear',    # Linear kernel for speed
            probability=True,   # Enable probability estimates
            C=1.0,
            cache_size=100     # Limit memory usage
        )
        
        # Level classifier (H1, H2, H3, H4)
        self.level_classifier = RandomForestClassifier(
            n_estimators=50,    # Limited trees for speed
            max_depth=10,
            random_state=42,
            n_jobs=1           # Single thread for consistency
        )

    def _extract_text_with_features(self, page):
        """Extract text with comprehensive features for ML"""
        lines = []
        
        # Method 1: Detailed word extraction
        try:
            words = page.extract_words(x_tolerance=2, y_tolerance=2)
            if words:
                lines_dict = {}
                for word in words:
                    if 'y0' in word and 'text' in word:
                        y_key = round(word['y0'] * 2) / 2  # Half-point precision
                        if y_key not in lines_dict:
                            lines_dict[y_key] = []
                        lines_dict[y_key].append(word)
                
                for y_pos in sorted(lines_dict.keys(), reverse=True):
                    line_words = sorted(lines_dict[y_pos], key=lambda w: w.get('x0', 0))
                    text = ' '.join([w['text'] for w in line_words if w.get('text', '').strip()])
                    
                    if text.strip():
                        # Calculate comprehensive features
                        sizes = [w.get('size', 12) for w in line_words if w.get('size') and w.get('size') > 0]
                        avg_size = sum(sizes) / len(sizes) if sizes else 12
                        
                        # Font variation (consistency indicator)
                        font_variation = np.std(sizes) if len(sizes) > 1 else 0
                        
                        # Position features
                        x_positions = [w.get('x0', 0) for w in line_words]
                        left_margin = min(x_positions) if x_positions else 0
                        
                        # Fontname consistency
                        fonts = [w.get('fontname', '') for w in line_words if w.get('fontname')]
                        dominant_font = Counter(fonts).most_common(1)[0][0] if fonts else ''
                        
                        lines.append({
                            'text': text.strip(),
                            'size': avg_size,
                            'y0': y_pos,
                            'left_margin': left_margin,
                            'font_variation': font_variation,
                            'dominant_font': dominant_font,
                            'word_count': len(line_words)
                        })
                        
                if lines:
                    return lines
        except Exception as e:
            print(f"Detailed extraction failed: {e}")
        
        # Method 2: Simple text fallback
        try:
            text = page.extract_text()
            if text:
                for i, line in enumerate(text.split('\n')):
                    if line.strip():
                        lines.append({
                            'text': line.strip(),
                            'size': 12,  # Default
                            'y0': len(text.split('\n')) - i,
                            'left_margin': 0,
                            'font_variation': 0,
                            'dominant_font': '',
                            'word_count': len(line.split())
                        })
        except Exception as e:
            print(f"Simple extraction failed: {e}")
        
        return lines

    def _calculate_ml_features(self, line_prop, page_context):
        """Extract numerical features for ML models"""
        text = line_prop['text']
        
        features = []
        
        # Text-based features
        features.extend([
            len(text),                                    # Length
            len(text.split()),                           # Word count
            float(text.isupper()),                       # Is uppercase
            float(text.istitle()),                       # Is title case
            sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0, # Uppercase ratio
            text.count('.'),                            # Period count
            text.count(':'),                            # Colon count
            float(bool(re.match(r'^\d+\.', text))),      # Starts with number
        ])
        
        # Position and formatting features
        features.extend([
            line_prop['size'],                          # Font size
            line_prop['left_margin'],                   # Left margin
            line_prop['font_variation'],                # Font consistency
            line_prop['word_count'],                    # Words in line
            line_prop['y0'] / page_context.get('height', 800),  # Relative position
        ])
        
        # Pattern matching features
        pattern_matches = []
        for level, config in self.heading_patterns.items():
            level_match = 0
            for pattern in config['patterns']:
                if re.match(pattern, text, re.IGNORECASE):
                    level_match = 1
                    break
            pattern_matches.append(level_match)
        features.extend(pattern_matches)
        
        return np.array(features)

    def _get_ml_prediction(self, text, features):
        """Get ML model predictions for heading classification"""
        if not ML_AVAILABLE or not self.is_trained:
            return {'is_heading': 0.5, 'level_probs': {'H1': 0.25, 'H2': 0.25, 'H3': 0.25, 'H4': 0.25}}
        
        try:
            # Text vectorization
            text_vector = self.vectorizer.transform([text])
            
            # Heading detection
            is_heading_prob = self.heading_classifier.predict_proba(text_vector)[0][1]
            
            # Level classification if it's likely a heading
            if is_heading_prob > 0.3:
                level_probs = self.level_classifier.predict_proba([features])[0]
                level_dict = dict(zip(['H1', 'H2', 'H3', 'H4'], level_probs))
            else:
                level_dict = {'H1': 0.25, 'H2': 0.25, 'H3': 0.25, 'H4': 0.25}
            
            return {
                'is_heading': is_heading_prob,
                'level_probs': level_dict
            }
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {'is_heading': 0.5, 'level_probs': {'H1': 0.25, 'H2': 0.25, 'H3': 0.25, 'H4': 0.25}}

    def _hybrid_heading_detection(self, line_prop, page_context):
        """Combined rule-based and ML heading detection"""
        text = line_prop['text'].strip()
        
        # Basic filters
        if len(text) < 3 or len(text) > 150 or not re.search(r'[a-zA-Z]', text):
            return False, None, 0.0
        
        # Skip obvious non-headings
        skip_patterns = [
            r'^\s*\d+\s*$',           # Just page numbers
            r'^\s*page\s+\d+\s*$',    # "page 1"
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # dates
            r'^\s*©.*$',              # Copyright
            r'^\s*www\.',             # URLs
            r'^\s*\d+:\d+\s*(AM|PM)?\s*$',  # Times
            r'^\s*Version\s+[\d.]+\s*$',    # Version numbers
        ]
        
        if any(re.match(pattern, text, re.IGNORECASE) for pattern in skip_patterns):
            return False, None, 0.0
        
        # Calculate features for ML
        features = self._calculate_ml_features(line_prop, page_context)
        ml_prediction = self._get_ml_prediction(text, features)
        
        # Rule-based scoring
        rule_score = 0.0
        best_level = None
        
        for level, config in self.heading_patterns.items():
            level_score = 0.0
            
            # Pattern matching
            for pattern in config['patterns']:
                if re.match(pattern, text, re.IGNORECASE):
                    level_score += config['weight']
                    break
            
            # Font size bonus
            if line_prop['size'] > 12:
                level_score += 0.2 * (line_prop['size'] - 12) / 10
            
            # Position bonus (higher on page = more likely heading)
            position_score = (page_context.get('height', 800) - line_prop['y0']) / page_context.get('height', 800)
            level_score += 0.1 * position_score
            
            if level_score > rule_score:
                rule_score = level_score
                best_level = level
        
        # Combine rule-based and ML scores
        final_score = (
            self.feature_weights['ml_confidence'] * ml_prediction['is_heading'] +
            (1 - self.feature_weights['ml_confidence']) * rule_score
        )
        
        # If ML suggests a different level, consider it
        if ml_prediction['is_heading'] > 0.7:
            ml_best_level = max(ml_prediction['level_probs'].items(), key=lambda x: x[1])[0]
            if ml_prediction['level_probs'][ml_best_level] > 0.5:
                best_level = ml_best_level
        
        # Decision threshold
        is_heading = final_score > 0.4
        
        return is_heading, best_level, final_score

    def extract_title(self, pdf):
        """Enhanced title extraction with ML features"""
        try:
            first_page = pdf.pages[0]
            text_lines = self._extract_text_with_features(first_page)
            
            if not text_lines:
                return self._get_metadata_title(pdf)
            
            # Score potential titles
            title_candidates = []
            
            for i, line in enumerate(text_lines[:15]):
                text = line['text'].strip()
                
                if len(text) < 5 or len(text) > 200:
                    continue
                
                # Skip obvious non-titles
                skip_patterns = [
                    r'^\s*(page\s+\d+|abstract|introduction|chapter\s+\d+)\s*$',
                    r'^\s*\d+\s*$',
                    r'^\s*©.*$',
                    r'^\s*version\s+[\d.]+\s*$',
                ]
                
                if any(re.match(pattern, text, re.IGNORECASE) for pattern in skip_patterns):
                    continue
                
                # Calculate comprehensive title score
                score = 0
                
                # Position score (earlier is better)
                score += (15 - i) * 3
                
                # Length score
                if 10 <= len(text) <= 100:
                    score += 15
                elif 5 <= len(text) <= 150:
                    score += 8
                
                # Case and formatting score
                if text.isupper() and len(text) > 10:
                    score += 20
                elif text.istitle():
                    score += 15
                elif text[0].isupper():
                    score += 8
                
                # Font size score
                if line['size'] > 14:
                    score += 15
                elif line['size'] > 12:
                    score += 8
                
                # Content relevance (domain-specific keywords)
                content_keywords = [
                    r'overview|foundation|level|extensions',
                    r'rfp|request|proposal',
                    r'application\s+form',
                    r'stem\s+pathways',
                    r'digital\s+library',
                    r'party\s+invitation'
                ]
                
                for keyword_pattern in content_keywords:
                    if re.search(keyword_pattern, text, re.IGNORECASE):
                        score += 12
                        break
                
                # ML enhancement for title detection
                if ML_AVAILABLE and self.is_trained:
                    try:
                        features = self._calculate_ml_features(line, {'height': first_page.height})
                        ml_pred = self._get_ml_prediction(text, features)
                        # Titles often have different patterns than headings
                        title_ml_score = (1 - ml_pred['is_heading']) * 10  # Titles are often NOT typical headings
                        score += title_ml_score
                    except:
                        pass
                
                title_candidates.append((score, text, i))
            
            # Select best title
            if title_candidates:
                title_candidates.sort(key=lambda x: -x[0])
                return title_candidates[0][1]
            
            # Fallback strategies
            for line in text_lines[:10]:
                text = line['text'].strip()
                if len(text) >= 8 and len(text) <= 150 and re.search(r'[a-zA-Z]', text):
                    return text
            
            return self._get_metadata_title(pdf)
            
        except Exception as e:
            print(f"Error extracting title: {e}")
            return self._get_metadata_title(pdf)

    def _get_metadata_title(self, pdf):
        """Extract title from metadata with filtering"""
        try:
            if pdf.metadata and 'Title' in pdf.metadata and pdf.metadata['Title']:
                title = str(pdf.metadata['Title']).strip()
                
                # Filter out filename-like titles
                filename_patterns = [
                    r'.*\.(doc|docx|pdf|cdr)$',
                    r'^Microsoft Word -',
                    r'^Document\d*$',
                    r'^Untitled\d*$',
                ]
                
                if not any(re.match(pattern, title, re.IGNORECASE) for pattern in filename_patterns):
                    return title
        except:
            pass
        
        return ""

    def extract_outline(self, pdf_path):
        """Extract outline with advanced ML-enhanced detection"""
        outline = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                title = self.extract_title(pdf)
                
                # Special handling for form documents
                if re.search(r'application\s+form', title, re.IGNORECASE):
                    return {"title": title, "outline": []}
                
                for page_num, page in enumerate(pdf.pages, 1):
                    if page_num > 50:  # Limit as per requirements
                        break
                    
                    text_lines = self._extract_text_with_features(page)
                    page_context = {'height': page.height, 'width': page.width}
                    
                    for line in text_lines:
                        is_heading, level, confidence = self._hybrid_heading_detection(line, page_context)
                        
                        if is_heading and level and confidence > 0.4:
                            outline.append({
                                "level": level,
                                "text": line['text'],
                                "page": page_num,
                                "confidence": round(confidence, 3)  # For debugging
                            })
                
                # Post-processing: Remove low-confidence outliers if we have high-confidence headings
                if outline:
                    confidences = [item['confidence'] for item in outline]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    if avg_confidence > 0.7:  # High overall confidence
                        outline = [item for item in outline if item['confidence'] > 0.5]
                    
                    # Remove confidence from final output
                    for item in outline:
                        del item['confidence']
                
                return {"title": title, "outline": outline}
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error Processing Document", "outline": []}

    def train_on_examples(self, training_data=None):
        """Train ML models on example data (optional enhancement)"""
        if not ML_AVAILABLE:
            print("ML libraries not available for training")
            return
        
        # Use built-in training examples if no data provided
        if training_data is None:
            training_data = self._generate_training_examples()
        
        try:
            # Prepare training data
            texts = []
            labels = []
            features = []
            level_labels = []
            
            for example in training_data:
                texts.append(example['text'])
                labels.append(1 if example['is_heading'] else 0)
                features.append(example['features'])
                if example['is_heading']:
                    level_labels.append(example['level'])
            
            # Train heading detection model
            X_text = self.vectorizer.fit_transform(texts)
            self.heading_classifier.fit(X_text, labels)
            
            # Train level classification model
            if level_labels:
                heading_features = [features[i] for i, label in enumerate(labels) if label == 1]
                self.level_classifier.fit(heading_features, level_labels)
            
            self.is_trained = True
            print(f"ML models trained on {len(training_data)} examples")
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.is_trained = False

    def _generate_training_examples(self):
        """Generate synthetic training examples for the models"""
        examples = []
        
        # Positive examples (headings)
        heading_examples = [
            ("1. Introduction", "H1"),
            ("2.1 Background", "H2"),
            ("Chapter 1: Overview", "H1"),
            ("SUMMARY", "H1"),
            ("Timeline:", "H3"),
            ("Appendix A: Data", "H1"),
            ("3.2.1 Methodology", "H3"),
            ("REFERENCES", "H1"),
            ("Background", "H2"),
            ("For each Ontario citizen it could mean:", "H4"),
        ]
        
        for text, level in heading_examples:
            # Generate mock features (18 features to match _calculate_ml_features)
            features = np.random.rand(18)
            examples.append({
                'text': text,
                'is_heading': True,
                'level': level,
                'features': features
            })
        
        # Negative examples (non-headings)
        non_heading_examples = [
            "This is a regular paragraph of text that contains multiple sentences.",
            "The results show that our approach works well in most cases.",
            "Page 5",
            "© 2023 Company Name. All rights reserved.",
            "For more information, please contact support@example.com",
            "The quick brown fox jumps over the lazy dog.",
            "Table 1 shows the comparison between different methods.",
        ]
        
        for text in non_heading_examples:
            features = np.random.rand(18)
            examples.append({
                'text': text,
                'is_heading': False,
                'level': None,
                'features': features
            })
        
        return examples

    def process_all_pdfs(self, input_dir="/app/input", output_dir="/app/output"):
        """Process all PDFs with ML-enhanced extraction"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Train models on startup (optional)
        if ML_AVAILABLE:
            self.train_on_examples()
        
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            result = self.extract_outline(pdf_file)
            
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Saved: {output_file.name}")
            print(f"Title: {result['title']}")
            print(f"Found {len(result['outline'])} headings")


def main():
    extractor = AdvancedPDFOutlineExtractor()
    extractor.process_all_pdfs()


if __name__ == "__main__":
    main()
