import time, json, os, re, hashlib, csv, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "ingest_sources_config.json"
LOG_PATH = BASE / "crawler_log.csv"
OUT_ARCH = BASE / "architecture" / "processed.jsonl"
OUT_STRUCT = BASE / "structure" / "processed.jsonl"
OUT_URBAN = BASE / "urban_planning" / "processed.jsonl"
VOCAB_OUT = BASE / "vocab" / "auto_extracted_terms.json"

MIN_BODY_LEN = 200

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_text(text: str) -> str:
    # Basic Persian normalization
    replacements = {
        '\u064A': 'ی',  # Arabic Yeh
        '\u0643': 'ک',  # Arabic Kaf
    }
    for a,b in replacements.items():
        text = text.replace(a,b)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_terms(body: str):
    # Very naive term extraction (frequency-based for Persian words >4 chars)
    tokens = re.findall(r'[\u0600-\u06FF]{4,}', body)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t,0)+1
    return {k:v for k,v in freq.items() if v>1}

def get_html(url: str):
    if not requests:
        raise RuntimeError("requests not installed")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
    except Exception:
        return None
    return None

def parse_article(url: str, source: Dict[str,Any]):
    html = get_html(url)
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser') if BeautifulSoup else None
    if not soup:
        return None
    title = soup.title.string.strip() if soup.title else url
    # Collect paragraph text
    paras = [p.get_text(' ', strip=True) for p in soup.find_all('p')]
    body = normalize_text(' '.join(paras))
    if len(body) < MIN_BODY_LEN:
        return None
    doc_id = f"{source['name']}_{hashlib.sha256(url.encode()).hexdigest()[:10]}"
    return {
        "id": doc_id,
        "source": source['name'],
        "url": url,
        "lang": "fa",
        "title": title,
        "tags": [],
        "body": body,
        "created": datetime.utcnow().isoformat()+'Z'
    }

def write_jsonl(path: Path, doc: Dict[str,Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

def log_event(row: Dict[str,Any]):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","source","url","status","note"])
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def crawl_source(source: Dict[str,Any]):
    if not source.get('allowed', False):
        log_event({"timestamp": datetime.utcnow().isoformat(), "source": source['name'], "url": source['base_url'], "status": "skip", "note": "only_metadata"})
        return
    base = source['base_url']
    html = get_html(base)
    if not html:
        log_event({"timestamp": datetime.utcnow().isoformat(), "source": source['name'], "url": base, "status": "error", "note": "fetch_failed"})
        return
    soup = BeautifulSoup(html, 'html.parser') if BeautifulSoup else None
    if not soup:
        return
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('#'): continue
        if href.startswith('javascript'): continue
        full = urljoin(base, href)
        # limit domain
        if urlparse(full).netloc != urlparse(base).netloc:
            continue
        links.append(full)
    # Deduplicate & limit
    unique = list(dict.fromkeys(links))[:15]
    for link in unique:
        art = parse_article(link, source)
        if art:
            target = OUT_ARCH if source['type'] in ['projects','architecture','case-study'] else OUT_URBAN
            if source['type'] == 'forum':
                target = OUT_STRUCT  # forum engineering discussions
            write_jsonl(target, art)
            log_event({"timestamp": datetime.utcnow().isoformat(), "source": source['name'], "url": link, "status": "stored", "note": f"len={len(art['body'])}"})
        else:
            log_event({"timestamp": datetime.utcnow().isoformat(), "source": source['name'], "url": link, "status": "skip", "note": "too_short_or_fail"})
        time.sleep(cfg['rate_limit_seconds'])

def build_vocab_auto():
    buckets = [OUT_ARCH, OUT_STRUCT, OUT_URBAN]
    aggregate = {}
    for b in buckets:
        if not b.exists():
            continue
        with open(b, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    terms = extract_terms(obj.get('body',''))
                    for k,v in terms.items():
                        aggregate[k] = aggregate.get(k,0)+v
                except Exception:
                    continue
    # Keep top 200
    top = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)[:200]
    vocab_list = [{"term": t, "freq": f} for t,f in top]
    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)
    return vocab_list

if __name__ == '__main__':
    cfg = load_config()
    for src in cfg['sources']:
        crawl_source(src)
    auto_vocab = build_vocab_auto()
    print(f"Auto vocabulary size: {len(auto_vocab)}")
    print("Ingestion complete.")
