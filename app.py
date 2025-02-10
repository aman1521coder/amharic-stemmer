from flask import Flask, render_template, request
from demo import AmharicIRSystem  # Ensure correct import

app = Flask(__name__)

# Initialize Amharic IR System
ir_system = AmharicIRSystem(
    doc_dir="/home/oem/Documents/document_ir",
    stopword_file="/home/oem/Desktop/Amharic-rule-based-lemmatizer-master/amharic_stopwords.txt",
    lexicon_file="dictionary/amh_lex_dic.trans.txt"
)
@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        
        # Example: Predefined relevant document IDs for the query
        relevant_docs = [1, 2, 3]  # Replace with actual relevant document IDs
        
        # Perform search and calculate metrics
        results = ir_system.search(query, relevant_docs=relevant_docs)
        metrics = results.get('metrics', {})  # Get metrics from results

        # Print metrics for debugging
        print("Metrics:", metrics)

        return render_template('results.html', 
                               results=results['results'],
                               metrics=metrics,
                               query=query)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
