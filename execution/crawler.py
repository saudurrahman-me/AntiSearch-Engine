import sqlite3
import urllib.robotparser
import urllib.parse
import urllib.request
import requests
from bs4 import BeautifulSoup
import time
import datetime
from urllib.parse import urlparse, urljoin, urlunparse
import json
import os

from execution.config import settings

# Load Configuration
CONFIG_PATH = "d:/Studies/SearchEngine/config.json"
DB_PATH = settings.DB_PATH

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SEED_URLS = config.get("seed_urls", [])
ALLOWED_DOMAINS = set(config.get("allowed_domains", []))
MAX_PAGES = config.get("max_pages", 1000)
MAX_DEPTH = config.get("max_depth", 10)
DELAY = config.get("crawl_delay_seconds", 0.5)
TIMEOUT = config.get("request_timeout", 10)

# Global Robots Cache
robots_cache = {}

def setup_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            headings TEXT,
            content TEXT,
            crawled_at TIMESTAMP,
            depth INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS links (
            source_url TEXT,
            target_url TEXT,
            PRIMARY KEY (source_url, target_url)
        )
    ''')
    conn.commit()
    return conn

def normalize_url(url):
    """Normalize URL by removing fragments, query parameters, and trailing slashes."""
    parsed = urlparse(url)
    # Strip trailing slash from path unless it's the root
    path = parsed.path
    if path != '/' and path.endswith('/'):
        path = path[:-1]
    
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        '', # params
        '', # query
        ''  # fragment
    ))
    return normalized

def is_allowed_by_robots(url, user_agent="CustomBot/1.0"):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in robots_cache:
        robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            robots_cache[domain] = rp
        except Exception as e:
            robots_cache[domain] = None
            
    rp = robots_cache[domain]
    if rp is None:
        return True # Default to allow if robots.txt failed
    
    return rp.can_fetch(user_agent, url)

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def get_requests_session():
    session = requests.Session()
    # Retry 5 times, backoff factor 0.5 (0.5s, 1s, 2s, 4s, 8s) for typical network/DNS errors
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[ 500, 502, 503, 504 ], 
                    allowed_methods=["GET"])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def crawl():
    conn = setup_db()
    cursor = conn.cursor()
    
    frontier = [(normalize_url(url), 0) for url in SEED_URLS]
    visited = set()
    crawled_count = 0
    errors = 0
    skipped_duplicates = 0
    
    session = get_requests_session()
    
    # Check already visited to resume
    cursor.execute("SELECT url FROM pages")
    for row in cursor.fetchall():
        visited.add(row[0])
        crawled_count += 1
        
    print(f"Starting crawler. Already visited {len(visited)} pages. Target: {MAX_PAGES}")
    
    # Using a set to quickly check frontier contents to avoid slow O(N) list search
    frontier_urls = {u for u, d in frontier}
    
    # Process frontier
    while frontier and crawled_count < MAX_PAGES:
        current_url, depth = frontier.pop(0)
        if current_url in frontier_urls:
            frontier_urls.remove(current_url)
        
        if current_url in visited:
            skipped_duplicates += 1
            continue
            
        parsed = urlparse(current_url)
        if parsed.netloc not in ALLOWED_DOMAINS:
            continue
            
        if depth > MAX_DEPTH:
            continue
            
        if not is_allowed_by_robots(current_url):
            print(f"[{crawled_count}/{MAX_PAGES}] Blocked by robots.txt: {current_url}")
            visited.add(current_url)
            continue
            
        print(f"[{crawled_count}/{MAX_PAGES}] [Depth {depth}] Crawling: {current_url}")
        
        try:
            response = session.get(current_url, headers={"User-Agent": "CustomBot/1.0"}, timeout=TIMEOUT)
            if response.status_code != 200:
                print(f"  -> Failed (status: {response.status_code})")
                errors += 1
                visited.add(current_url)
                continue
                
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                visited.add(current_url)
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
            
            # Extract headings
            headings_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            headings = ' '.join([h.get_text(separator=' ').strip() for h in headings_tags])
            
            # Extract main text
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.extract()
            text = soup.get_text(separator=' ')
            text = ' '.join(text.split())
            
            if not text:
                print("  -> Empty content, skipping.")
                errors += 1
                visited.add(current_url)
                continue
            
            # Save page to DB
            crawled_at = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO pages (url, title, headings, content, crawled_at, depth)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (current_url, title, headings, text, crawled_at, depth))
            
            # Extract links
            links_extracted = 0
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(current_url, href)
                norm_target = normalize_url(absolute_url)
                
                parsed_target = urlparse(norm_target)
                if parsed_target.scheme in ['http', 'https'] and parsed_target.netloc in ALLOWED_DOMAINS:
                    links_extracted += 1
                    
                    # Save link edges for PageRank
                    cursor.execute('''
                        INSERT OR IGNORE INTO links (source_url, target_url)
                        VALUES (?, ?)
                    ''', (current_url, norm_target))
                    
                    if norm_target not in visited and norm_target not in frontier_urls:
                        frontier.append((norm_target, depth + 1))
                        frontier_urls.add(norm_target)
                            
            conn.commit()
            visited.add(current_url)
            crawled_count += 1
            print(f"  -> Saved. Found {links_extracted} internal links.")
            
            time.sleep(DELAY)
            
        except requests.exceptions.RequestException as e:
            print(f"  -> Network Error: {e}")
            errors += 1
            visited.add(current_url)
        except Exception as e:
            print(f"  -> Parsing Error: {e}")
            errors += 1
            visited.add(current_url)

    print("\n--- Crawl Statistics ---")
    print(f"Total Pages Crawled: {crawled_count}")
    print(f"Total Errors: {errors}")
    print(f"Skipped Duplicates: {skipped_duplicates}")
    print(f"Remaining in Frontier: {len(frontier)}")
    conn.close()

if __name__ == "__main__":
    crawl()
