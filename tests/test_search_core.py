import pytest
from execution.search_core import search, get_vstore

def test_get_vstore_singleton():
    """Verify that get_vstore returns the same instance to cache the model."""
    vstore1 = get_vstore()
    vstore2 = get_vstore()
    assert vstore1 is vstore2

def test_search_bm25_empty():
    """Verify search handles empty or meaningless queries gracefully."""
    results = search("xxxxxxxxxxxnonexistentxxxxxxxxxx", mode="bm25")
    assert len(results) == 0

def test_search_hybrid_valid():
    """Verify hybrid search runs and returns expected data structures."""
    # Ensure warmup to not count loading time
    get_vstore()
    results = search("python array", mode="hybrid", limit=5)
    
    assert isinstance(results, list)
    if len(results) > 0:
        res = results[0]
        assert "doc_id" in res
        assert "title" in res
        assert "score" in res
        assert "url" in res
        assert "snippet" in res
        assert res["score"] > 0
