import os
import re
import math
from collections import defaultdict
from docx import Document
from flask import Flask, render_template, request

app = Flask(__name__)

# Configuration
DOC_DIR = "/home/oem/Documents/document_ir"
STOPWORD_FILE = "/home/oem/Desktop/Amharic-rule-based-lemmatizer-master/amharic_stopwords.txt"
LEXICON_FILE = "dictionary/amh_lex_dic.trans.txt"

# Load documents and build index
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
    return text.lower()

def load_stopwords(stopword_file):
    if os.path.exists(stopword_file):
        with open(stopword_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    else:
        print(f"Warning: Stopword file '{stopword_file}' not found. Continuing without stopwords.")
        return set()

def load_lexical_data(lexicon_file):
    if os.path.exists(lexicon_file):
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Warning: Lexicon file '{lexicon_file}' not found. Stemming accuracy may be affected.")
        return ""

def stem(input1, lexical_data):
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

def disambiguate(stems, lexical_data):
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

def get_stem(token, lexical_data):
    stems = stem(token, lexical_data)
    best_match = disambiguate(stems, lexical_data)
    return best_match.group(1) if best_match else token

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
        
        doc_length = len(stems)
        doc_lengths[doc_id] = doc_length
        total_terms += doc_length

        term_freq = defaultdict(int)
        for stemmed in stems:
            term_freq[stemmed] += 1
            
        for term, freq in term_freq.items():
            inverted_index[term].append((doc_id, freq))
            document_frequencies[term] += 1

    avgdl = total_terms / len(documents) if documents else 0
            
    return inverted_index, document_frequencies, doc_lengths, avgdl

# Metric evaluation (Precision, Recall, F1, MAP, NDCG, MRR)
def calculate_metrics(retrieved_docs, relevant_docs):
    metrics = {}
    relevant_set = set(relevant_docs)
    
    # Precision, Recall, F1 calculation
    for k in [1, 3, 5, 10, len(retrieved_docs)]:
        retrieved_at_k = set(retrieved_docs[:k])
        tp = len(retrieved_at_k & relevant_set)
        
        precision = tp / k if k != 0 else 0
        recall = tp / len(relevant_set) if len(relevant_set) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics[f'P@{k}'] = precision
        metrics[f'R@{k}'] = recall
        metrics[f'F1@{k}'] = f1

    # MAP: Mean Average Precision
    ap_scores = []
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            relevant_count = sum([1 for j in range(i) if retrieved_docs[j] in relevant_set])
            precision_at_i = relevant_count / i
            ap_scores.append(precision_at_i)
    map_score = sum(ap_scores) / len(relevant_set) if relevant_set else 0
    metrics['MAP'] = map_score

    # NDCG: Normalized Discounted Cumulative Gain
    ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(len(relevant_set))])
    dcg = sum([(1.0 if doc_id in relevant_set else 0) / math.log2(i + 2) 
             for i, doc_id in enumerate(retrieved_docs)])
    ndcg = dcg / ideal_dcg if ideal_dcg else 0
    metrics['NDCG'] = ndcg

    # MRR: Mean Reciprocal Rank
    mrr = 0
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            mrr = 1.0 / i
            break
    metrics['MRR'] = mrr

    return metrics

def process_query(query, inverted_index, document_frequencies, doc_lengths, avgdl, stopwords, lexical_data, relevant_docs=None, k1=1.5, b=0.75):
    normalized = normalize_text(query)
    tokens = tokenize_amharic(normalized)
    filtered = [t for t in tokens if t not in stopwords]
    query_terms = [get_stem(t, lexical_data) for t in filtered]

    scores = defaultdict(float)
    
  
    for term in query_terms:
        if term in inverted_index:
            N = len(doc_lengths)
            df = document_frequencies[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            for (doc_id, tf) in inverted_index[term]:
                doc_len = doc_lengths[doc_id]
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                scores[doc_id] += idf * tf_component

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    retrieved_docs = [doc_id for doc_id, _ in ranked_results]
    
    metrics = {}
    if relevant_docs:
        metrics = calculate_metrics(retrieved_docs, relevant_docs)

    return ranked_results, metrics

# Initialize system
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

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        relevant_docs = {
        "የመንግሥት ፕሮጀክቶች": [2, 30, 65, 86],
        "የ ኢትዮጲያ ፖለቲካ": [9, 12, 13, 15, 21, 24, 45, 75, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 95, 97, 99, 101],
        "የአርሶ አደር ልማት": [3, 64],
        "እድገት በኢትዮጵያ": [5, 11, 25, 30, 33, 36, 40, 42, 72, 73, 74, 77, 93, 101],
        "መብት እና እኩልነት": [2, 4, 6, 7, 10, 13, 20, 34, 41, 43, 68, 70, 81, 92],
        "የኢትዮጵያ ስፖርት": list(range(46, 63)),
        "የኢትዮጵያ ኢኮኖሚ": [1, 2, 5, 8, 11, 13, 14, 16, 17, 22, 25, 28, 29, 31, 32, 33, 35, 37, 40, 42, 69, 71, 74, 79, 91, 93, 94, 98, 100, 101],
        "የአማራ ክልል": [7, 15, 22, 41, 67, 82],
        "ኢትዮጵያ ውስጥ በሚካሄዱ ጦርነቶች": [4, 7, 12, 15, 20, 27, 76, 78, 84, 85, 96],
        "የአስተዳደር ማሻሻያ እና ለውጥ": [16, 26, 29, 31, 65, 81, 86],
        "የጤና መረጃ": [18, 19, 38, 74],
        "የህግ የበላይነት": [15, 32, 35, 36, 39, 44, 76, 78, 85],
        "ትምህርትና ምርምር": [63, 66, 67]
    } 
        
        results, metrics = process_query(
            query, 
            inverted_index, 
            doc_freq,
            doc_lengths,
            avgdl,
            stopwords, 
            lexical_data,
            relevant_docs
        )
        
        return render_template('results.html',
                              results=results[:10],  # Show top 10 results
                              metrics=metrics,
                              query=query,
                              documents=documents)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
