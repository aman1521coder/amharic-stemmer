import re
import json
import math
from collections import defaultdict
from docx import Document
import os
class AmharicIRSystem:
    def __init__(self, doc_dir, stopword_file, lexicon_file):
        self.doc_dir = doc_dir
        self.stopword_file = stopword_file
        self.lexicon_file = lexicon_file
        self.documents = {}
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = {}
        self.avgdl = 0
        self.stopwords = set()
        self.lexical_data = ""
        
        self._initialize_system()

    def _initialize_system(self):
        self._load_linguistic_resources()
        self._process_documents()
        self._build_inverted_index()

    def _load_linguistic_resources(self):
        # Load stopwords
        if os.path.exists(self.stopword_file):
            with open(self.stopword_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
        
        # Load lexical data
        if os.path.exists(self.lexicon_file):
            with open(self.lexicon_file, 'r', encoding='utf-8') as f:
                self.lexical_data = f.read()

    def _process_documents(self):
        for idx, filename in enumerate(os.listdir(self.doc_dir)):
            if filename.endswith(".docx"):
                doc_path = os.path.join(self.doc_dir, filename)
                doc = Document(doc_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                self.documents[idx + 1] = text

    def _build_inverted_index(self):
        total_terms = 0
        for doc_id, text in self.documents.items():
            normalized = self._normalize_text(text)
            tokens = self._tokenize_amharic(normalized)
            filtered = [t for t in tokens if t not in self.stopwords]
            stems = [self._get_stem(t) for t in filtered]
            
            self.doc_lengths[doc_id] = len(stems)
            total_terms += len(stems)

            term_freq = defaultdict(int)
            for stemmed in stems:
                term_freq[stemmed] += 1
                
            for term, freq in term_freq.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freq[term] += 1

        self.avgdl = total_terms / len(self.documents) if self.documents else 0

    def _tokenize_amharic(self, text):
        return re.findall(r'[\u1200-\u137F\uAB00-\uAB2F]+', text)

    def _normalize_text(self, text):
        return text.lower()

    def _get_stem(self, token):
        stems = self._stem(token)
        best_match = self._disambiguate(stems)
        return best_match.group(1) if best_match else token

    def _stem(self, input1):
        collection = [input1]

        # Suffix stripping rules
        suffix_patterns = [
            r"(.+)(iwu|wu|wi|awī|na|mi|ma|li|ne|ache)$",
            r"(.+)(ochi|bache|wache|chi|ku|ki|ache|wal)$",
            r"(.+)(iwu|wu|w|awī|na|mi|ma|li|ne|che)$",
            r"(.+)(ochi|bache|wache|chi|ku|ki|che|wal)$"
        ]
        
        # Prefix stripping rules
        prefix_patterns = [
            r"^(yete|inide|inidī|āli)(.+)$",
            r"^(ye|yi|masi|le|ke|inid|be|sile)(.+)$",
            r"^(te|mī|mi|me|mayit|ma|bale|yit)(.+)$"
        ]

        # Apply suffix rules
        current = input1
        for pattern in suffix_patterns:
            match = re.match(pattern, current)
            if match:
                current = match.group(1)
                collection.append(current)

        # Apply prefix rules
        current = input1
        for pattern in prefix_patterns:
            match = re.match(pattern, current)
            if match:
                current = match.group(2)
                collection.append(current)

        return list(set(collection))  # Return unique stems

    def _disambiguate(self, stems):
        match = None
        string_size = 0
        for stem_candidate in stems:
            temp = re.search(f'({re.escape(stem_candidate)}) {{.+}}', self.lexical_data)
            if temp:
                if len(stem_candidate) > string_size:
                    string_size = len(stem_candidate)
                    match = temp
            else:
                modified = re.sub(r'(.+)[īaou]\b', r'\1i', stem_candidate)
                if modified != stem_candidate:
                    stems.append(modified)
        return match

    def search(self, query, relevant_docs=None, k1=1.2, b=0.75):
        normalized = self._normalize_text(query)
        tokens = self._tokenize_amharic(normalized)
        filtered = [t for t in tokens if t not in self.stopwords]
        query_terms = [self._get_stem(t) for t in filtered]

        scores = defaultdict(float)
        metrics = {}

        for term in query_terms:
            if term in self.inverted_index:
                N = len(self.doc_lengths)
                df = self.doc_freq[term]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                
                for (doc_id, tf) in self.inverted_index[term]:
                    doc_len = self.doc_lengths[doc_id]
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avgdl))
                    scores[doc_id] += idf * tf_component

        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if relevant_docs:
            metrics = self._calculate_metrics(ranked_results, relevant_docs)

        return {
            'results': ranked_results[:100],  # Return top 100 results
            'metrics': metrics,
            'query_terms': query_terms
        }

    def _calculate_metrics(self, ranked_results, relevant_docs):
        metrics = {}
        retrieved_docs = [doc_id for doc_id, _ in ranked_results]
        relevant_set = set(relevant_docs)
        
        # Precision-Recall Curve data
        precision_points = []
        recall_points = []
        
        for k in range(1, len(retrieved_docs)+1):
            retrieved_at_k = set(retrieved_docs[:k])
            tp = len(retrieved_at_k & relevant_set)
            
            precision = tp / k if k != 0 else 0
            recall = tp / len(relevant_set) if len(relevant_set) != 0 else 0
            
            precision_points.append(precision)
            recall_points.append(recall)

        # Standard metrics
        for k in [5, 10, 20]:
            retrieved_at_k = set(retrieved_docs[:k])
            tp = len(retrieved_at_k & relevant_set)
            
            metrics[f'P@{k}'] = tp / k
            metrics[f'R@{k}'] = tp / len(relevant_set)

        # Mean Average Precision
        ap_scores = []
        relevant_count = 0
        precision_sum = 0
        
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / i
                
        metrics['MAP'] = precision_sum / len(relevant_set) if relevant_set else 0

        # NDCG
        ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(len(relevant_set))])
        dcg = sum([(1.0 if doc_id in relevant_set else 0) / math.log2(i + 2) 
                 for i, doc_id in enumerate(retrieved_docs)])
        metrics['NDCG'] = dcg / ideal_dcg if ideal_dcg else 0

        # ROC Curve data
        return {
            'precision_recall': {
                'precision': precision_points,
                'recall': recall_points
            },
            'standard_metrics': metrics,
            'ndcg': metrics['NDCG']
        }