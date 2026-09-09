import { useState, useEffect } from 'react'
import './index.css'

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [results, setResults] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setHasSearched(true)
    try {
      const response = await fetch(`http://localhost:8000/api/search?q=${encodeURIComponent(query)}&mode=${mode}`)
      const data = await response.json()
      setResults(data.results)
      setStats({
        total: data.total_results,
        time: data.execution_time_ms.toFixed(2),
        mode: data.mode
      })
    } catch (error) {
      console.error("Search failed:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={`app-container ${!hasSearched ? 'centered' : ''}`}>
      {!hasSearched && (
        <div className="logo">
          Anti<span>Search</span>
        </div>
      )}
      
      <div className="search-container">
        <form onSubmit={handleSearch}>
          <div className="search-input-group">
            <input
              type="text"
              className="search-input"
              placeholder="Search the web..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
            <select 
              className="mode-selector" 
              value={mode} 
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="bm25">Keyword (BM25)</option>
              <option value="semantic">Semantic (AI)</option>
              <option value="hybrid">Hybrid (Best)</option>
            </select>
          </div>
        </form>
      </div>

      {loading && <div className="loader"></div>}

      {hasSearched && !loading && stats && (
        <>
          <div className="search-stats">
            Found {stats.total} results in {stats.time} ms (Mode: <span className="stats-mode">{stats.mode.toUpperCase()}</span>)
          </div>
          
          <div className="results-container">
            {results.map((result, index) => (
              <div 
                className="result-card" 
                key={result.doc_id}
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <a href={result.url} className="result-url" target="_blank" rel="noreferrer">
                  {result.url}
                </a>
                <a href={result.url} className="result-title" target="_blank" rel="noreferrer">
                  {result.title}
                </a>
                <p 
                  className="result-snippet"
                  dangerouslySetInnerHTML={{ __html: result.snippet }}
                ></p>
                <div className="result-metrics">
                  <span>Score: {result.score.toFixed(3)}</span>
                  <span>BM25: {result.bm25_score.toFixed(3)}</span>
                  <span>PR: {result.pagerank.toFixed(5)}</span>
                </div>
              </div>
            ))}
            
            {results.length === 0 && (
              <div style={{ color: 'var(--text-secondary)' }}>
                No documents match your query. Try different keywords.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default App
