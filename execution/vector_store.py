import os
import json
import sqlite3

from execution.config import settings

class VectorStore:
    def __init__(self, db_path=None):
        self.db_path = db_path or settings.DB_PATH
        self.faiss_path = settings.FAISS_PATH
        self.index = None
        self.model = None
        self.doc_map = []
        self.dimension = settings.EMBEDDING_DIMENSION
        
    def _lazy_load(self, load_existing=True):
        if self.model is not None:
            return
            
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        import faiss
        if load_existing and os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            
            # Load doc_map from SQLite corpus_stats
            try:
                conn = sqlite3.connect(self.db_path, timeout=5)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM corpus_stats WHERE key = 'faiss_doc_map'")
                row = cursor.fetchone()
                if row:
                    self.doc_map = json.loads(row[0])
                conn.close()
            except sqlite3.OperationalError:
                self.doc_map = []
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.doc_map = []

    def reset(self):
        self._lazy_load(load_existing=False)

    def encode(self, text_list):
        self._lazy_load()
        return self.model.encode(text_list, normalize_embeddings=True)
        
    def add_vectors(self, vectors, doc_ids):
        self._lazy_load()
        import numpy as np
        self.index.add(np.array(vectors))
        self.doc_map.extend(doc_ids)
        
    def save(self):
        self._lazy_load()
        import faiss
        faiss.write_index(self.index, self.faiss_path)
        
        # Save mapping to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO corpus_stats (key, value) VALUES ('faiss_doc_map', ?)", (json.dumps(self.doc_map),))
        conn.commit()
        conn.close()

    def search(self, query, k=100):
        self._lazy_load()
        if self.index is None or self.index.ntotal == 0:
            return {}
            
        q_emb = self.encode([query])
        import numpy as np
        
        # Clamp k to number of available vectors
        k_search = min(k, self.index.ntotal)
        if k_search <= 0:
            return {}
            
        D, I = self.index.search(np.array(q_emb), k_search)
        
        results = {}
        for dist, row_idx in zip(D[0], I[0]):
            if 0 <= row_idx < len(self.doc_map):
                doc_id = self.doc_map[row_idx]
                results[doc_id] = float(dist)
                
        return results
