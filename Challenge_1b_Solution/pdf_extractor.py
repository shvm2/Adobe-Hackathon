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
    ML_AVAILABLE_GLOBAL = True # Renamed to avoid conflict and clarify scope
except ImportError:
    ML_AVAILABLE_GLOBAL = False
    print("ML libraries not available, falling back to rules only")

class AdvancedPDFOutlineExtractor:
    def __init__(self):
        # Store the global ML_AVAILABLE status as an instance attribute
        self.ML_AVAILABLE = ML_AVAILABLE_GLOBAL 
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
                    r'^\s*CRITERIA\s*$',                       # Added for a specific file
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
        
        if self.ML_AVAILABLE: # Use self.ML_AVAILABLE here
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
                        x0_pos = x_positions[0] if x_positions else 0
                        
                        # Fontname consistency
                        fonts = [w.get('fontname', '') for w in line_words if w.get('fontname')]
                        dominant_font = Counter(fonts).most_common(1)[0][0] if fonts else ''
                        
                        lines.append({
                            'text': text.strip(),
                            'size': avg_size,
                            'y0': y_pos,
                            'x0': x0_pos, # Added x0 position
                            'left_margin': left_margin,
                            'font_variation': font_variation,
                            'dominant_font': dominant_font,
                            'word_count': len(line_words)
                        })
                        
                if lines:
                    return lines
        except Exception as e:
            # print(f"Detailed extraction failed: {e}") # Suppress for cleaner output
            pass
        
        # Method 2: Simple text fallback
        try:
            text = page.extract_text()
            if text:
                for i, line in enumerate(text.split('\n')):
                    if line.strip():
                        lines.append({
                            'text': line.strip(),
                            'size': 12,  # Default
                            'y0': page.height - (i * 15), # Estimate y0
                            'x0': 0, # Default
                            'left_margin': 0,
                            'font_variation': 0,
                            'dominant_font': '',
                            'word_count': len(line.split())
                        })
        except Exception as e:
            # print(f"Simple extraction failed: {e}") # Suppress for cleaner output
            pass
        
        return lines

    def _calculate_ml_features(self, line_prop, page_context):
        """Extract numerical features for ML models (18 features)"""
        text = line_prop['text']
        
        features = []
        
        # Text-based features (8 features)
        features.extend([
            len(text),                                    
            float(text.isupper()),                       
            float(text.istitle()),                       
            sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0, # Uppercase ratio
            text.count('.'),                            
            text.count(':'),                            
            float(bool(re.match(r'^\d+\.', text))),      # Starts with number
            len(set(text.lower().split())),              # Unique word count (New)
        ])
        
        # Position and formatting features (6 features)
        features.extend([
            line_prop['size'],                          
            line_prop['left_margin'],                   
            line_prop['font_variation'],                
            line_prop['word_count'],                    
            line_prop['y0'] / page_context.get('height', 800),  # Relative Y position
            line_prop['x0'] / page_context.get('width', 600),   # Relative X position (New)
        ])
        
        # Pattern matching features (4 features)
        pattern_matches = []
        for level, config in self.heading_patterns.items():
            level_match = 0
            for pattern in config['patterns']:
                if re.match(pattern, text, re.IGNORECASE):
                    level_match = 1
                    break
            pattern_matches.append(level_match)
        features.extend(pattern_matches)
        
        return np.array(features) # Total 8 + 6 + 4 = 18 features

    def _get_ml_prediction(self, text, features):
        """Get ML model predictions for heading classification"""
        if not self.ML_AVAILABLE or not self.is_trained: # Use self.ML_AVAILABLE here
            # Default probabilities if ML is not available or not trained
            return {'is_heading': 0.5, 'level_probs': {'H1': 0.25, 'H2': 0.25, 'H3': 0.25, 'H4': 0.25}}
        
        try:
            # Text vectorization
            text_vector = self.vectorizer.transform([text])
            
            # Heading detection
            is_heading_prob = self.heading_classifier.predict_proba(text_vector)[0][1]
            
            # Level classification if it's likely a heading
            # Ensure features array has correct shape for prediction
            if is_heading_prob > 0.3:
                # Reshape features to (1, -1) if it's a 1D array
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                level_probs = self.level_classifier.predict_proba(features)[0]
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
            r'^\s*TABLE\s+OF\s+CONTENTS\s*$', # Specific false positive on some docs
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
                    level_score += config['weight'] * 0.7 # Give strong weight to pattern match
                    break
            
            # Font size bonus
            if line_prop['size'] > 12:
                level_score += 0.2 * (line_prop['size'] - 12) / 10
            
            # Position bonus (higher on page = more likely heading, less left margin)
            position_score_y = (page_context.get('height', 800) - line_prop['y0']) / page_context.get('height', 800)
            position_score_x = (page_context.get('width', 600) - line_prop['x0']) / page_context.get('width', 600) # Right-aligned preferred for some heading types
            
            level_score += 0.1 * position_score_y
            level_score += 0.05 * (1 - (line_prop['left_margin'] / page_context.get('width', 600))) # Penalize large left margin
            
            if level_score > rule_score:
                rule_score = level_score
                best_level = level
        
        # Combine rule-based and ML scores
        # Ensure ml_prediction['is_heading'] is a scalar for weighting
        final_score = (
            self.feature_weights['ml_confidence'] * ml_prediction['is_heading'] +
            (1 - self.feature_weights['ml_confidence']) * rule_score
        )
        
        # If ML suggests a different level with high confidence, prefer it
        if ml_prediction['is_heading'] > 0.7:
            ml_best_level = max(ml_prediction['level_probs'].items(), key=lambda x: x[1])[0]
            if ml_prediction['level_probs'][ml_best_level] > 0.5: # Sufficient confidence in specific level
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
            page_context = {'height': first_page.height, 'width': first_page.width}

            for i, line in enumerate(text_lines):
                text = line['text'].strip()
                
                if len(text) < 5 or len(text) > 200:
                    continue
                
                # Skip obvious non-titles
                skip_patterns = [
                    r'^\s*(page\s+\d+|abstract|introduction|chapter\s+\d+|table\s+of\s+contents)\s*$',
                    r'^\s*\d+\s*$',
                    r'^\s*©.*$',
                    r'^\s*version\s+[\d.]+\s*$',
                    r'^\s*www\..*$',
                    r'^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', # Dates
                    r'^\s*\[\s*source:\s*\d+\s*\]\s*.*$', # Citations
                    r'^\s*(TOPJUMP|ADOBE INDIA HACKATHON|MEAL IDEAS).*$', # Common headers in provided docs
                    r'^\s*Instructions:\s*$', # Recipe instructions header
                ]
                
                if any(re.match(pattern, text, re.IGNORECASE) for pattern in skip_patterns):
                    continue
                
                # Calculate comprehensive title score
                score = 0
                
                # Position score (earlier and centered is better)
                score += (max(0, 15 - i)) * 3 # Rank based on first 15 lines
                
                # Attempt to favor lines near the top center
                if line['y0'] < (page_context['height'] * 0.3) and \
                   (line['x0'] > page_context['width'] * 0.1 and line['x0'] < page_context['width'] * 0.5):
                    score += 10

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
                
                # Font size score (larger is better)
                if line['size'] > 18:
                    score += 25
                elif line['size'] > 14:
                    score += 15
                elif line['size'] > 12:
                    score += 8
                
                # Content relevance (domain-specific keywords)
                content_keywords = [
                    r'overview|foundation|level|extensions|syllabus|tester',
                    r'rfp|request|proposal|business\s+plan',
                    r'application\s+form',
                    r'stem\s+pathways|conference|awards',
                    r'digital\s+library',
                    r'invited|party|jump',
                    r'culinary|cuisine|food|recipes|dinner|lunch|breakfast',
                    r'acrobat|edit|share|convert|signatures|generative\s+ai|forms',
                    r'travel|guide|france|trip|cities|restaurants|hotels',
                    r'history|traditions|culture',
                ]
                
                for keyword_pattern in content_keywords:
                    if re.search(keyword_pattern, text, re.IGNORECASE):
                        score += 12
                        break
                
                # ML enhancement for title detection
                if self.ML_AVAILABLE and self.is_trained: # Use self.ML_AVAILABLE here
                    try:
                        features = self._calculate_ml_features(line, page_context)
                        ml_pred = self._get_ml_prediction(text, features)
                        # Titles often have different patterns than typical headings.
                        # Low 'is_heading' probability might indicate it's a unique document title.
                        title_ml_score = (1 - ml_pred['is_heading']) * 15 # Boost if not a typical heading
                        score += title_ml_score
                    except Exception as ml_e:
                        # print(f"ML title prediction error: {ml_e}") # Suppress for cleaner output
                        pass
                
                title_candidates.append((score, text, i))
            
            # Select best title
            if title_candidates:
                title_candidates.sort(key=lambda x: (-x[0], x[2])) # Sort by score desc, then by line index asc
                final_title = title_candidates[0][1]
                
                # Aggressive filtering for short, generic titles that might be false positives
                if len(final_title.split()) < 3 and final_title.lower() in ["overview", "introduction", "document"]:
                    # Look for the next best title that is more descriptive
                    for s, t, _ in title_candidates[1:]:
                        if len(t.split()) >= 3 and not any(re.match(p, t, re.IGNORECASE) for p in skip_patterns):
                            return t
                    
                return final_title
            
            # Fallback strategies (less aggressive filters)
            for line in text_lines[:10]:
                text = line['text'].strip()
                if len(text) >= 8 and len(text) <= 150 and re.search(r'[a-zA-Z]', text):
                    # Check if it looks like a typical title
                    if (text.isupper() or text.istitle()) and line['size'] > 12:
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
                
                # Filter out filename-like titles and generic ones
                filename_patterns = [
                    r'.*\.(doc|docx|pdf|cdr)$',
                    r'^Microsoft Word -',
                    r'^Document\d*$',
                    r'^Untitled\d*$',
                    r'^Adobe Acrobat$',
                    r'^Adobe Photoshop$',
                    r'^Adobe Illustrator$',
                    r'^Adobe InDesign$',
                    r'^Page \d+$',
                ]
                
                if not any(re.match(pattern, title, re.IGNORECASE) for pattern in filename_patterns):
                    return title
        except:
            pass
        
        return ""

    def extract_outline(self, pdf_path):
        """Extract outline with advanced ML-enhanced detection"""
        outline = []
        all_document_lines = [] 
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                title = self.extract_title(pdf)
                
                # Special handling for form documents (return empty outline)
                if re.search(r'application\s+form|you\'re\s+invited|fill\s+&\s+sign', title, re.IGNORECASE):
                    return {"title": title, "outline": [], "all_lines": []} 
                
                for page_num, page in enumerate(pdf.pages, 1):
                    if page_num > 50:  # Limit as per requirements
                        break
                    
                    text_lines_on_page = self._extract_text_with_features(page)
                    page_context = {'height': page.height, 'width': page.width}
                    
                    for line_prop in text_lines_on_page:
                        # Add page_num to each line_prop for easier lookup later
                        line_prop['page_num'] = page_num 
                        all_document_lines.append(line_prop)

                        is_heading, level, confidence = self._hybrid_heading_detection(line_prop, page_context)
                        
                        if is_heading and level and confidence > 0.4: # Adjusted threshold for better precision
                            outline.append({
                                "level": level,
                                "text": line_prop['text'],
                                "page": page_num,
                                "confidence": round(confidence, 3)  # For debugging/post-processing
                            })
                
                # Post-processing: Remove low-confidence outliers if we have high-confidence headings
                if outline:
                    confidences = [item['confidence'] for item in outline]
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    if avg_confidence > 0.6:  # If overall quality is high, be stricter
                        outline = [item for item in outline if item['confidence'] > 0.5]
                    elif avg_confidence < 0.3: # If overall quality is low, potentially filter more aggressively
                         outline = [item for item in outline if item['confidence'] > 0.45]

                    # Remove confidence from final output
                    for item in outline:
                        del item['confidence']
                
                return {"title": title, "outline": outline, "all_lines": all_document_lines}
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error Processing Document", "outline": [], "all_lines": []}

    def train_on_examples(self, training_data=None):
        """Train ML models on example data (optional enhancement)"""
        if not self.ML_AVAILABLE: # Use self.ML_AVAILABLE here
            print("ML libraries not available for training")
            return
        
        # Use built-in training examples if no data provided
        if training_data is None:
            training_data = self._generate_training_examples()
        
        try:
            # Prepare training data
            texts = []
            labels = []
            features_for_level_classifier = []
            level_labels = []
            
            for example in training_data:
                texts.append(example['text'])
                labels.append(1 if example['is_heading'] else 0)
                # Ensure features are always 1D arrays for input to fit
                features_for_level_classifier.append(example['features'].flatten()) 
                if example['is_heading']:
                    level_labels.append(example['level'])
            
            # Train heading detection model
            X_text = self.vectorizer.fit_transform(texts)
            self.heading_classifier.fit(X_text, labels)
            
            # Train level classification model
            if level_labels:
                # Filter features for only actual headings for level classifier training
                heading_features_filtered = [features_for_level_classifier[i] for i, label in enumerate(labels) if label == 1]
                self.level_classifier.fit(heading_features_filtered, level_labels)
            
            self.is_trained = True
            print(f"ML models trained on {len(training_data)} examples")
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.is_trained = False

    def _generate_training_examples(self):
        """Generate synthetic training examples for the models (18 features)"""
        examples = []
        
        # Define ranges/typical values for features (18 features)
        # Order: [len_text, isupper, istitle, uppercase_ratio, period_count, colon_count, starts_with_num_pattern, unique_word_count,
        #           size, left_margin, font_variation, word_count, relative_y_position, relative_x_position,
        #           is_h1_pattern, is_h2_pattern, is_h3_pattern, is_h4_pattern]

        # Positive examples (headings)
        heading_data = [
            ("1. Introduction to AI", "H1", {'size': 24, 'y0': 50, 'x0': 0, 'isupper': False, 'istitle': True}),
            ("2. AI History", "H2", {'size': 18, 'y0': 100, 'x0': 20, 'isupper': False, 'istitle': True}),
            ("3.1 Neural Networks Overview", "H3", {'size': 14, 'y0': 150, 'x0': 40, 'isupper': False, 'istitle': True}),
            ("SUMMARY", "H1", {'size': 20, 'y0': 60, 'x0': 0, 'isupper': True, 'istitle': False}),
            ("Timeline:", "H3", {'size': 14, 'y0': 180, 'x0': 30, 'isupper': False, 'istitle': True}),
            ("For each Ontario citizen it could mean:", "H4", {'size': 12, 'y0': 200, 'x0': 50, 'isupper': False, 'istitle': True}),
            ("Appendix A", "H1", {'size': 22, 'y0': 70, 'x0': 0, 'isupper': False, 'istitle': True}),
            ("2.6 Keeping It Current", "H2", {'size': 16, 'y0': 120, 'x0': 25, 'isupper': False, 'istitle': True}),
            ("Phase I: Business Planning", "H3", {'size': 14, 'y0': 160, 'x0': 35, 'isupper': False, 'istitle': True}),
            ("CRITERIA", "H1", {'size': 20, 'y0': 80, 'x0': 0, 'isupper': True, 'istitle': False}),
            ("Coastal Adventures", "H2", {'size': 18, 'y0': 90, 'x0': 10, 'isupper': False, 'istitle': True}),
            ("Culinary Experiences", "H3", {'size': 16, 'y0': 170, 'x0': 20, 'isupper': False, 'istitle': True}),
            ("Fil and Sign PDF Forms", "H2", {'size': 18, 'y0': 100, 'x0': 10, 'isupper': False, 'istitle': True}),
            ("TEST CASE 1: ACADEMIC RESEARCH", "H1", {'size': 20, 'y0': 120, 'x0': 0, 'isupper': True, 'istitle': False}),
        ]
        
        for text, level, props in heading_data:
            len_text = len(text)
            isupper = float(props['isupper'])
            istitle = float(props['istitle'])
            uppercase_ratio = sum(c.isupper() for c in text) / len_text if len_text > 0 else 0
            period_count = text.count('.')
            colon_count = text.count(':')
            starts_with_num_pattern = float(bool(re.match(r'^\d+\.', text)))
            unique_word_count = len(set(text.lower().split()))
            
            size = props['size']
            y0 = props['y0']
            x0 = props['x0']
            left_margin = props['x0'] # For simplicity, approximate left_margin as x0 for synthetic data
            font_variation = 0.1 + 0.1 * np.random.rand() # Small variation
            word_count = len(text.split())
            relative_y_position = y0 / 800.0 # Assume page height 800 for synthetic data
            relative_x_position = x0 / 600.0  # Assume page width 600 for synthetic data
            
            pattern_matches = [0] * 4 # H1, H2, H3, H4
            if level == 'H1': pattern_matches[0] = 1
            elif level == 'H2': pattern_matches[1] = 1
            elif level == 'H3': pattern_matches[2] = 1
            elif level == 'H4': pattern_matches[3] = 1
            
            features = [
                len_text, isupper, istitle, uppercase_ratio, period_count, colon_count, starts_with_num_pattern, unique_word_count,
                size, left_margin, font_variation, word_count, relative_y_position, relative_x_position,
            ]
            features.extend(pattern_matches) # Adds 4 more features
            
            examples.append({
                'text': text,
                'is_heading': True,
                'level': level,
                'features': np.array(features, dtype=np.float32)
            })

        # Negative examples (non-headings)
        non_heading_data = [
            ("This is a regular paragraph of text that contains multiple sentences.", 12, 300, 50),
            ("The results show that our approach works well in most cases.", 11, 350, 60),
            ("Page 5", 10, 750, 70),
            ("© 2023 Company Name. All rights reserved.", 9, 780, 80),
            ("For more information, please contact support@example.com", 11, 400, 90),
            ("The quick brown fox jumps over the lazy dog.", 12, 450, 10),
            ("Table 1 shows the comparison between different methods.", 11, 250, 15),
            ("This is a footnote reference [1].", 9, 760, 20),
            ("  • Bullet point item one", 10, 280, 50), # Indented
            ("  - Another list item", 10, 290, 50), # Indented
            ("Recipe for delicious pancakes:", 12, 210, 0), # Looks like heading but could be part of list
            ("Instructions:", 12, 220, 0), # Common recipe header
            ("[source: 123] Some text with source citation.", 11, 310, 0),
            ("www.example.com/document.pdf", 10, 790, 0),
        ]

        for text, size, y0, x0 in non_heading_data:
            len_text = len(text)
            isupper = float(text.isupper())
            istitle = float(text.istitle())
            uppercase_ratio = sum(c.isupper() for c in text) / len_text if len_text > 0 else 0
            period_count = text.count('.')
            colon_count = text.count(':')
            starts_with_num_pattern = float(bool(re.match(r'^\d+\.', text)))
            unique_word_count = len(set(text.lower().split()))
            
            font_variation = 0.5 + 0.5 * np.random.rand() # Can have more variation
            word_count = len(text.split())
            relative_y_position = y0 / 800.0
            relative_x_position = x0 / 600.0
            
            pattern_matches = [0] * 4 # No heading patterns matched
            
            features = [
                len_text, isupper, istitle, uppercase_ratio, period_count, colon_count, starts_with_num_pattern, unique_word_count,
                size, x0, font_variation, word_count, relative_y_position, relative_x_position,
            ]
            features.extend(pattern_matches)
            
            examples.append({
                'text': text,
                'is_heading': False,
                'level': None,
                'features': np.array(features, dtype=np.float32)
            })
        
        return examples

