import sqlite3
import math
import re
import os
import json
from execution.indexer import clean_text, DB_PATH

# Global VectorStore instance to cache the model in memory across API requests
_vstore_instance = None

def get_vstore():
    global _vstore_instance
    if _vstore_instance is None:
        from execution.vector_store import VectorStore
        _vstore_instance = VectorStore(DB_PATH)
    return _vstore_instance

# BM25 Parameters
k1 = 1.5
b = 0.75

# Vector search is handled modularly by VectorStore


def get_snippet(content, query_tokens, window=60):
    text = content
    text_lower = text.lower()
    
    # Find the first occurrence of any query token
    best_idx = -1
    best_token_len = 0
    for token in query_tokens:
        # Use regex to find whole word match for accurate boundary
        match = re.search(r'\b' + re.escape(token) + r'\b', text_lower)
        if match:
            idx = match.start()
            if best_idx == -1 or idx < best_idx:
                best_idx = idx
                best_token_len = len(token)
                
    if best_idx == -1:
        return content[:200] + "..."
        
    # Attempt to expand window to the nearest space
    start = max(0, best_idx - window)
    if start > 0:
        # snap to next space to avoid half-words
        space_idx = text.find(' ', start)
        if space_idx != -1 and space_idx < best_idx:
            start = space_idx + 1
            
    end = min(len(content), best_idx + best_token_len + window)
    if end < len(content):
        # snap to previous space
        space_idx = text.rfind(' ', best_idx, end)
        if space_idx != -1 and space_idx > best_idx + best_token_len:
            end = space_idx
            
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
        
    # Bold the matched words using word boundaries
    for token in query_tokens:
        pattern = re.compile(rf'\b({re.escape(token)})\b', re.IGNORECASE)
        snippet = pattern.sub(r"<b>\1</b>", snippet)
        
    return snippet


def search(query, mode="bm25", limit=10, pr_weight=0.1):
    """
    mode: "bm25", "semantic", "hybrid"
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get Corpus Stats
    cursor.execute("SELECT key, value FROM corpus_stats")
    stats = dict(cursor.fetchall())
    N = stats.get('total_docs', 0)
    avgdl = stats.get('avg_doc_length', 1)
    
    if N == 0:
        return []

    # Get PageRanks
    cursor.execute("SELECT doc_id, pagerank FROM document_stats")
    pageranks = dict(cursor.fetchall())
    
    # --- 1. BM25 Search ---
    bm25_scores = {}
    query_tokens = clean_text(query)
    if query_tokens and mode in ["bm25", "hybrid"]:
        # Pre-fetch all document lengths to avoid N+1 query problem
        cursor.execute("SELECT doc_id, length FROM document_stats")
        doc_lengths = dict(cursor.fetchall())
        
        for token in query_tokens:
            cursor.execute("SELECT df FROM df_index WHERE term = ?", (token,))
            row = cursor.fetchone()
            if not row: continue
            df = row[0]
            
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            cursor.execute("SELECT doc_id, tf FROM inverted_index WHERE term = ?", (token,))
            postings = cursor.fetchall()
            
            for doc_id, tf in postings:
                doc_len = doc_lengths.get(doc_id, avgdl)
                tf_bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                score = idf * tf_bm25
                bm25_scores[doc_id] = bm25_scores.get(doc_id, 0) + score

    # --- 2. Semantic Search ---
    semantic_scores = {}
    if mode in ["semantic", "hybrid"]:
        vstore = get_vstore()
        results = vstore.search(query, k=min(100, N))
        semantic_scores = results

    # --- 3. Ranking & Fusion ---
    final_scores = []
    
    if mode == "bm25":
        for doc_id, b_score in bm25_scores.items():
            pr = pageranks.get(doc_id, 0)
            expected_pr = 1 / max(N, 1)
            pr_factor = pr / expected_pr
            pr_boost = pr_weight * math.log(1 + pr_factor)
            final_scores.append((doc_id, b_score + pr_boost, b_score, pr))
            
    elif mode == "semantic":
        for doc_id, s_score in semantic_scores.items():
            pr = pageranks.get(doc_id, 0)
            expected_pr = 1 / max(N, 1)
            pr_factor = pr / expected_pr
            pr_boost = pr_weight * math.log(1 + pr_factor)
            final_scores.append((doc_id, s_score + pr_boost, s_score, pr))
            
    elif mode == "hybrid":
        # Reciprocal Rank Fusion (RRF)
        # 1. Rank BM25
        ranked_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        # 2. Rank Semantic
        ranked_sem = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        
        rrf_k = 60
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(ranked_bm25):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (rrf_k + rank + 1))
            
        for rank, (doc_id, _) in enumerate(ranked_sem):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (rrf_k + rank + 1))
            
        for doc_id, rrf_score in rrf_scores.items():
            pr = pageranks.get(doc_id, 0)
            expected_pr = 1 / max(N, 1)
            pr_factor = pr / expected_pr
            pr_boost = pr_weight * math.log(1 + pr_factor)
            # RRF is usually very small (e.g., 0.03), scale up for display / PR addition
            scaled_rrf = rrf_score * 100 
            final_scores.append((doc_id, scaled_rrf + pr_boost, scaled_rrf, pr))
            
    # Sort and take top K
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = final_scores[:limit]
    
    results = []
    if top_k:
        top_ids = [item[0] for item in top_k]
        placeholders = ','.join(['?'] * len(top_ids))
        
        cursor.execute(f"SELECT id, url, title, content FROM pages WHERE id IN ({placeholders})", top_ids)
        docs_info = {row[0]: {'url': row[1], 'title': row[2], 'content': row[3]} for row in cursor.fetchall()}
        
        for doc_id, score, base_score, pr in top_k:
            info = docs_info.get(doc_id)
            if info:
                results.append({
                    'doc_id': doc_id,
                    'url': info['url'],
                    'title': info['title'],
                    'snippet': get_snippet(info['content'], query_tokens),
                    'score': score,
                    'bm25_score': base_score,
                    'pagerank': pr
                })
                
    conn.close()
    return results
