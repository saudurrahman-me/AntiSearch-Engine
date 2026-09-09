# AntiSearch Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![React](https://img.shields.io/badge/react-18-blue.svg)
![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)

AntiSearch is a fully-featured, portfolio-ready search engine built from the ground up to explore modern information retrieval techniques. It blends traditional keyword algorithms (BM25) with state-of-the-art AI semantic embeddings and graph-based authority scoring (PageRank) into a single, unified Hybrid Search pipeline.

## Features

- **Hybrid Ranking Algorithm**: Combines Exact Keyword matching (BM25), Semantic Context matching (SentenceTransformers/FAISS), and Authority (PageRank) using Reciprocal Rank Fusion (RRF).
- **Asynchronous Web Crawler**: Fast, respectful (robots.txt compliant), depth-limited crawling of documentation sites directly into an inverted index.
- **Lightning Fast Vector Search**: Employs `faiss-cpu` for sub-50ms vector dot-product similarity lookups across thousands of documents.
- **Glassmorphism React UI**: A stunning, modern, and highly responsive frontend to toggle between search algorithms instantly.
- **Dockerized**: Backend and Frontend are containerized with a `docker-compose.yml` for unified deployment.

## Architecture

```mermaid
flowchart TD
    subgraph Offline Pipeline
        A["Web Crawler (crawler.py)"] -->|Stores HTML & Links| B[("(SQLite Database)")]
        B -->|Computes| C["Inverted Index (indexer.py)"]
        B -->|Calculates| D["PageRank Scores"]
        B -->|Generates Embeddings| E[("(FAISS Vector Index)")]
    end

    subgraph Online Pipeline
        F["React UI (App.jsx)"] -->|GET /api/search| G["FastAPI Backend (api.py)"]
        G -->|BM25 Query| B
        G -->|Semantic Query (vector_store.py)| E
        G -->|RRF Fusion (search_core.py)| G
        G -->|JSON Results| F
    end
```

### Core Modules
- `execution/config.py`: Environment-based `pydantic` configuration loader.
- `execution/crawler.py`: Fetches and extracts text from external URLs.
- `execution/indexer.py`: Computes BM25 tokens, constructs the `inverted_index`, computes PageRank, and initializes the FAISS embedding vectors.
- `execution/search_core.py`: Evaluates and fuses the scoring modes.
- `execution/vector_store.py`: A memory-cached singleton class wrapping PyTorch `SentenceTransformers` and `faiss-cpu`.
- `execution/api.py`: FastAPI server exposing the search functionality securely to the frontend.

## Setup & Installation

### Option 1: Native Windows Setup (Tested & Verified)

To run this application naturally on Windows without Docker:

1. **Backend Setup:**
   ```powershell
   # Copy the environment template
   Copy-Item .env.example .env

   # Create and activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
2. **Crawl & Index Data:** *(Skip if `data/` already contains the `.db` and `.faiss` files)*
   ```powershell
   python -m execution.crawler
   python -m execution.indexer
   ```
3. **Start the API:**
   ```powershell
   python -m execution.api
   ```
4. **Start the Frontend:**
   ```powershell
   # Open a new terminal
   cd frontend
   npm install
   npm run dev
   ```

### Option 2: Docker (Experimental)

> **Note on Docker Support**: A `docker-compose.yml`, `Dockerfile.backend`, and `Dockerfile.frontend` are provided for ease of use in Linux/Mac environments. However, these images have not been locally verified because Docker Desktop is not currently installed on the primary Windows development machine.

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```
2. **Start the containers:**
   ```bash
   docker-compose up --build -d
   ```
3. **Access the application:**
   - Frontend UI: `http://localhost:3000`
   - API Docs: `http://localhost:8000/docs`

## Benchmark Results & Testing

A robust test suite validating BM25 exact matches, Semantic conceptual mapping, and FastAPI input validation routing is included. **All 7 automated Pytest checks pass successfully.**

During an extensive 15-query evaluation suite (`benchmark.py`) over the Python 3.14 Documentation database, our AI semantic and hybrid pipelines significantly outperformed standard exact-keyword lookups on conceptual searches:

| Metric | BM25 (Keyword) | Semantic (AI) | Hybrid (BM25 + Semantic) |
|--------|----------------|---------------|--------------------------|
| **Latency** | 75.4 ms | 28.9 ms | 30.5 ms |
| **Accuracy (Exact)** | High | Moderate | High |
| **Accuracy (Conceptual)**| Low | High | High |

*Semantic search is consistently faster than traditional database-level BM25 due to in-memory highly-optimized C++ FAISS vector operations.*

## API Usage Examples

```powershell
# Perform a hybrid search
Invoke-RestMethod -Uri "http://localhost:8000/api/search?q=how%20to%20save%20data&mode=hybrid"

# Check system health
Invoke-RestMethod -Uri "http://localhost:8000/api/health"
```
