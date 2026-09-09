import sqlite3
import re
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Need to ensure nltk data is downloaded (run during setup)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from execution.config import settings

DB_PATH = settings.DB_PATH

# Global instances to avoid repeated initialization
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
# Force WordNet to load into memory now, avoiding a 3-second delay on the first query
LEMMATIZER.lemmatize("warmup")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    
    cleaned_tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS and len(token) > 1]
    return cleaned_tokens

def calculate_pagerank(conn, num_iterations=20, d=0.85):
    cursor = conn.cursor()
    print("Calculating PageRank...")
    
    # Get all pages
    cursor.execute("SELECT url FROM pages")
    pages = [row[0] for row in cursor.fetchall()]
    url_to_index = {url: i for i, url in enumerate(pages)}
    N = len(pages)
    
    if N == 0:
        return {}
        
    # Build adjacency matrix
    out_links = {i: [] for i in range(N)}
    cursor.execute("SELECT source_url, target_url FROM links")
    for src, tgt in cursor.fetchall():
        if src in url_to_index and tgt in url_to_index:
            out_links[url_to_index[src]].append(url_to_index[tgt])
            
    # PageRank algorithm
    pr = np.ones(N) / N
    for iteration in range(num_iterations):
        new_pr = np.zeros(N)
        for i in range(N):
            if len(out_links[i]) > 0:
                share = pr[i] / len(out_links[i])
                for j in out_links[i]:
                    new_pr[j] += share
            else:
                # Dangling node (distribute its PageRank evenly)
                new_pr += pr[i] / N
                
        # Add teleportation factor
        pr = (1 - d) / N + d * new_pr
        
    url_pagerank = {pages[i]: pr[i] for i in range(N)}
    print("PageRank calculation complete.")
    return url_pagerank

def build_index():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables for indexing
    cursor.executescript('''
        DROP TABLE IF EXISTS inverted_index;
        CREATE TABLE inverted_index (
            term TEXT,
            doc_id INTEGER,
            tf INTEGER,
            PRIMARY KEY (term, doc_id)
        );
        
        DROP TABLE IF EXISTS document_stats;
        CREATE TABLE document_stats (
            doc_id INTEGER PRIMARY KEY,
            length INTEGER,
            pagerank REAL
        );
        
        DROP TABLE IF EXISTS corpus_stats;
        CREATE TABLE corpus_stats (
            key TEXT PRIMARY KEY,
            value REAL
        );
        
        DROP TABLE IF EXISTS df_index;
        CREATE TABLE df_index (
            term TEXT PRIMARY KEY,
            df INTEGER
        );
        
        CREATE INDEX IF NOT EXISTS idx_term ON inverted_index(term);
    ''')
    
    # Calculate PageRank
    pageranks = calculate_pagerank(conn)
    
    cursor.execute("SELECT id, url, title, headings, content FROM pages")
    documents = cursor.fetchall()
    
    N = len(documents)
    total_length = 0
    df_counts = {}
    
    print(f"Indexing {N} documents (TF & Embeddings)...")
    
    # Track mapping of row index in FAISS to SQLite doc_id
    doc_ids = []
    texts_to_embed = []
    
    for doc_id, url, title, headings, content in documents:
        doc_ids.append(doc_id)
        
        # Tokenize fields separately for keyword BM25
        title_tokens = clean_text(title or "")
        heading_tokens = clean_text(headings or "")
        body_tokens = clean_text(content or "")
        
        # Prepare text for embedding: title + headings + up to 150 cleaned body words.
        # This gives a highly dense semantic representation of the document without noise.
        clean_body = " ".join(body_tokens[:150])
        semantic_text = f"{title}. {headings}. {clean_body}"
        texts_to_embed.append(semantic_text)
        
        # We consider the total document length as the sum of all tokens (for BM25 doc_len normalization)
        all_tokens = title_tokens + heading_tokens + body_tokens
        length = len(all_tokens)
        total_length += length
        
        # Calculate Weighted Term Frequency (TF)
        tf_counts = {}
        for token in title_tokens:
            tf_counts[token] = tf_counts.get(token, 0) + 3.0  # Title weight
        for token in heading_tokens:
            tf_counts[token] = tf_counts.get(token, 0) + 2.0  # Heading weight
        for token in body_tokens:
            tf_counts[token] = tf_counts.get(token, 0) + 1.0  # Body text weight
            
        # Update Document Frequency (DF) based on presence anywhere
        for term in set(all_tokens):
            df_counts[term] = df_counts.get(term, 0) + 1
            
        # Save to inverted_index
        for term, tf in tf_counts.items():
            cursor.execute('''
                INSERT INTO inverted_index (term, doc_id, tf)
                VALUES (?, ?, ?)
            ''', (term, doc_id, tf))
            
        # Save to document_stats
        pr = pageranks.get(url, 1.0/max(N, 1))
        cursor.execute('''
            INSERT INTO document_stats (doc_id, length, pagerank)
            VALUES (?, ?, ?)
        ''', (doc_id, length, pr))
        
    avg_length = total_length / max(N, 1)
    
    # Save df_index
    for term, df in df_counts.items():
        cursor.execute('''
            INSERT INTO df_index (term, df)
            VALUES (?, ?)
        ''', (term, df))
        
    # Save corpus_stats
    cursor.execute("INSERT INTO corpus_stats (key, value) VALUES ('total_docs', ?)", (N,))
    cursor.execute("INSERT INTO corpus_stats (key, value) VALUES ('avg_doc_length', ?)", (avg_length,))
    
    from execution.vector_store import VectorStore
    vstore = VectorStore(DB_PATH)
    vstore.reset()
    
    print("Computing embeddings in batches...")
    batch_size = 64
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        batch_ids = doc_ids[i:i+batch_size]
        
        embeddings = vstore.encode(batch)
        vstore.add_vectors(embeddings, batch_ids)
        print(f"  -> Embedded {min(i+batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")
        
    conn.commit()
    conn.close()
    
    vstore.save()
    
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()
