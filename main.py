import sys
import os
import re
import json
import math
from docx import Document
from collections import defaultdict

# --------------------------
# Stemmer Components
# --------------------------

def transliterate(user_input):
    trans_file = open('dictionary/transliterator/transliterator.min.json', 'r', encoding='utf-8')
    trans_data = json.loads(trans_file.read())
    trans_file.close()
    trans_data = eval(trans_data)

    characters = re.findall('\w', user_input)

    for char in characters:
        if char in trans_data:
            user_input = re.sub(char, trans_data[char], user_input)

    return user_input

def stem(input1):
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

def disambuigate(stems, lexical_data):
    match = None
    string_size = 0
    for stem_candidate in stems:
        temp = re.search(f'({re.escape(stem_candidate)}) {{.+}}', lexical_data)
        if temp:
            if len(stem_candidate) > string_size:
                string_size = len(stem_candidate)
                match = temp
        else:
            modified = re.sub(r'(.+)[īaou]\b', r'\1i', stem_candidate)
            if modified != stem_candidate:
                stems.append(modified)
    return match

# --------------------------
# IR System Components
# --------------------------

def process_documents(doc_directory):
    documents = {}
    for idx, filename in enumerate(os.listdir(doc_directory)):
        if filename.endswith(".docx"):
            doc_path = os.path.join(doc_directory, filename)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents[idx + 1] = text  # Using document IDs starting from 1
    return documents

def tokenize_amharic(text):
    return re.findall(r'[\u1200-\u137F\uAB00-\uAB2F]+', text)

def normalize_text(text):
    return text.lower()  # Amharic doesn't have case, but for other possible characters

def load_stopwords(stopword_file='amharic_stopwords.txt'):
    if os.path.exists(stopword_file):
        with open(stopword_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    else:
        print(f"Warning: Stopword file '{stopword_file}' not found. Continuing without stopwords.")
        return set()

def load_lexical_data(lexicon_file='dictionary/amh_lex_dic.trans.txt'):
    if os.path.exists(lexicon_file):
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Warning: Lexicon file '{lexicon_file}' not found. Stemming accuracy may be affected.")
        return ""

def get_stem(token, lexical_data):
    stems = stem(token)
    best_match = disambuigate(stems, lexical_data)
    return best_match.group(1) if best_match else token
# ... [Keep all previous imports and helper functions] ...

def build_inverted_index(documents, stopwords, lexical_data):
    inverted_index = defaultdict(list)
    document_frequencies = defaultdict(int)
    doc_lengths = {}
    total_terms = 0
    
    for doc_id, text in documents.items():
        normalized = normalize_text(text)
        tokens = tokenize_amharic(normalized)
        filtered = [t for t in tokens if t not in stopwords]
        stems = [get_stem(t, lexical_data) for t in filtered]
        
        # Store document length
        doc_length = len(stems)
        doc_lengths[doc_id] = doc_length
        total_terms += doc_length

        # Update term frequencies and inverted index
        term_freq = defaultdict(int)
        for stemmed in stems:
            term_freq[stemmed] += 1
            
        for term, freq in term_freq.items():
            inverted_index[term].append((doc_id, freq))
            document_frequencies[term] += 1

    # Calculate average document length
    avgdl = total_terms / len(documents) if documents else 0
            
    return inverted_index, document_frequencies, doc_lengths, avgdl

def process_query(query, inverted_index, document_frequencies, doc_lengths, avgdl, 
                 stopwords, lexical_data, relevant_docs=None, k1=1.2, b=0.75):
    # Preprocess query
    normalized = normalize_text(query)
    tokens = tokenize_amharic(normalized)
    filtered = [t for t in tokens if t not in stopwords]
    query_terms = [get_stem(t, lexical_data) for t in filtered]

    scores = defaultdict(float)
    
    for term in query_terms:
        if term in inverted_index:
            # Calculate BM25 IDF
            N = len(doc_lengths)
            df = document_frequencies[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # Calculate term weight for each document
            for (doc_id, tf) in inverted_index[term]:
                doc_len = doc_lengths[doc_id]
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                scores[doc_id] += idf * tf_component

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Enhanced evaluation metrics
    if relevant_docs:
        retrieved_docs = [doc_id for doc_id, _ in ranked_results]
        relevant_set = set(relevant_docs)
        
        # Precision and Recall at different levels
        metrics = {}
        for k in [5, 10, len(retrieved_docs)]:
            retrieved_at_k = set(retrieved_docs[:k])
            tp = len(retrieved_at_k & relevant_set)
            
            precision = tp / k if k != 0 else 0
            recall = tp / len(relevant_set) if len(relevant_set) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            
            metrics[f'P@{k}'] = precision
            metrics[f'R@{k}'] = recall
            metrics[f'F1@{k}'] = f1

        # Mean Average Precision (MAP)
        ap_scores = []
        relevant_count = 0
        precision_sum = 0
        
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / i
        map_score = precision_sum / len(relevant_set) if relevant_set else 0
        metrics['MAP'] = map_score

        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    return ranked_results

def main():
    # Configuration
    DOC_DIR = "/home/oem/Documents/document_ir"
    STOPWORD_FILE = "/home/oem/Desktop/Amharic-rule-based-lemmatizer-master/amharic_stopwords.txt"
    LEXICON_FILE = "dictionary/amh_lex_dic.trans.txt"
    
    # Initialize components
    print("Loading documents...")
    documents = process_documents(DOC_DIR)
    print(f"Loaded {len(documents)} documents")
    
    print("\nLoading linguistic resources...")
    stopwords = load_stopwords(STOPWORD_FILE)
    lexical_data = load_lexical_data(LEXICON_FILE)
    
    print("\nBuilding index...")
    inverted_index, doc_freq, doc_lengths, avgdl = build_inverted_index(
        documents, stopwords, lexical_data
    )
    
    # Query interface
    while True:
        query = input("\nEnter your Amharic search query (q to quit): ")
        if query.lower() == 'q':
            break
            
        results = process_query(
            query, 
            inverted_index, 
            doc_freq,
            doc_lengths,
            avgdl,
            stopwords, 
            lexical_data
        )
        
        print(f"\nFound {len(results)} results:")
        for doc_id, score in results[:10]:  # Show top 10 results
            print(f"Document {doc_id} (score: {score:.4f})")
            print(f"Content: {documents[doc_id][:200]}...\n")

if __name__ == "__main__":
    main()