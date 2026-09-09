import time
import statistics
import sys
import os

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.search_core import search, get_vstore

queries = {
    "Exact Keyword": [
        "sys.argv",
        "urllib.request",
        "json.loads",
        "class MyClass:"
    ],
    "Vocabulary Mismatch": [
        "make program run faster",
        "how to save data to disk",
        "catch errors and bugs",
        "talking to the internet"
    ],
    "Technical/Conceptual": [
        "object oriented programming",
        "asynchronous concurrent execution",
        "database connection",
        "regular expressions"
    ],
    "Multi-term": [
        "read csv file pandas",
        "parse html web scraping",
        "unit testing mock objects"
    ]
}

def evaluate():
    with open("final_evaluation_results.md", "w", encoding="utf-8") as f:
        f.write("# Final Search Quality Evaluation\n\n")
        
        # Warm up the models
        get_vstore()
        search("warmup", mode="semantic")
        
        latencies = {"bm25": [], "semantic": [], "hybrid": []}
        
        for category, cat_queries in queries.items():
            f.write(f"## {category} Queries\n\n")
            
            for q in cat_queries:
                f.write(f"### Query: `{q}`\n\n")
                
                for mode in ["bm25", "semantic", "hybrid"]:
                    start = time.time()
                    results = search(q, mode=mode, limit=3)
                    elapsed = (time.time() - start) * 1000
                    latencies[mode].append(elapsed)
                    
                    f.write(f"**{mode.upper()}** ({elapsed:.1f} ms):\n")
                    if not results:
                        f.write("- *No results found*\n")
                    else:
                        for i, r in enumerate(results):
                            if mode == "bm25":
                                score_str = f"Score: {r['score']:.2f}"
                            elif mode == "semantic":
                                score_str = f"Sim: {r['score']:.2f}"
                            else:
                                score_str = f"RRF: {r['score']:.2f}"
                            f.write(f"{i+1}. {r['title']} ({score_str})\n")
                    f.write("\n")
                f.write("---\n\n")
                
        f.write("## Average Warm-Query Latency\n\n")
        for mode, lats in latencies.items():
            avg = statistics.mean(lats)
            f.write(f"- **{mode.upper()}**: {avg:.1f} ms\n")

if __name__ == "__main__":
    evaluate()
