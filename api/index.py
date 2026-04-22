import glob
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, make_response, send_from_directory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import lexicon

ANALYZER = SentimentIntensityAnalyzer()
lexicon.install(ANALYZER)

USER_AGENT = "Mozilla/5.0 (compatible; VenturesSentimentBot/1.0; +https://ventures-portfolio-sentiment.vercel.app)"
FETCH_TIMEOUT = 8
MAX_WORKERS = 64
MAX_PER_SOURCE = 40

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "companies.json")
HISTORY_DIR = os.path.join(ROOT_DIR, "data", "history")

app = Flask(__name__, static_folder=ROOT_DIR, static_url_path="")

# Per-region Google News locales. Each locale = (hl, gl, ceid_lang).
# Multiple locales are fetched in parallel and results merged/deduped by URL.
REGION_LOCALES = {
    "India":     [("en-IN", "IN", "en")],
    "LatAm":     [("pt-BR", "BR", "pt-BR"), ("es-419", "MX", "es-419")],
    "EU":        [("en-GB", "GB", "en"), ("en-US", "US", "en")],
    "US":        [("en-US", "US", "en")],
    "GCC":       [("ar",     "AE", "ar"), ("en-US", "US", "en")],
    "Australia": [("en-AU", "AU", "en")],
    "SE Asia":   [("en-SG", "SG", "en")],
    "Other":     [("en-US", "US", "en")],
}


def load_companies():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["companies"]


def query_for(company):
    return company.get("search") or f'"{company["name"]}"'


def http_get(url, timeout=FETCH_TIMEOUT, user_agent=USER_AGENT):
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _google_news(query_str, hl, gl, ceid_lang, origin="news"):
    q = urllib.parse.quote(query_str)
    url = f"https://news.google.com/rss/search?q={q}+when:365d&hl={hl}&gl={gl}&ceid={gl}:{ceid_lang}"
    try:
        root = ET.fromstring(http_get(url))
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        src = item.find("source")
        items.append({
            "origin": origin,
            "title": title,
            "source": (src.text.strip() if (src is not None and src.text) else "Google News"),
            "link": (item.findtext("link") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
            "engagement": 0,
            "locale": hl,
        })
        if len(items) >= MAX_PER_SOURCE:
            break
    return items


def build_news_tasks(company):
    """Return (fn, args) pairs for every news fetch this company needs."""
    tasks = []
    query = query_for(company)
    domain = company.get("website", "")
    locales = REGION_LOCALES.get(company["region"], REGION_LOCALES["Other"])

    # Name-based query in each regional locale
    for (hl, gl, ceid) in locales:
        tasks.append(("news", (query, hl, gl, ceid)))

    # Domain-based query in en-US (catches URL-linked articles globally)
    if domain:
        tasks.append(("news", (f'"{domain}"', "en-US", "US", "en")))

    tasks.append(("hn", None))
    return tasks


def fetch_task(kind, args, company):
    if kind == "news":
        return _google_news(*args)
    if kind == "hn":
        return fetch_hn(company)
    return []


def fetch_hn(company):
    q = urllib.parse.quote(query_for(company))
    url = f"https://hn.algolia.com/api/v1/search?query={q}&tags=story&hitsPerPage={MAX_PER_SOURCE}"
    try:
        data = json.loads(http_get(url))
    except Exception:
        return []
    items = []
    for hit in data.get("hits", []):
        title = (hit.get("title") or hit.get("story_title") or "").strip()
        if not title:
            continue
        points = int(hit.get("points") or 0)
        comments = int(hit.get("num_comments") or 0)
        items.append({
            "origin": "hn",
            "title": title,
            "source": f"HN · {hit.get('author', '')}",
            "link": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}",
            "pubDate": hit.get("created_at", ""),
            "engagement": points + comments,
            "locale": "en",
        })
    return items


def score_company(company, items):
    base = {
        "name": company["name"],
        "website": company["website"],
        "region": company["region"],
        "description": company.get("description", ""),
        "count": 0, "score": 0.0,
        "pos": 0, "neu": 0, "neg": 0,
        "by_source": {"news": 0, "hn": 0},
        "by_locale": {},
        "engagement": 0,
        "last_seen": None,
        "headlines": [],
    }
    if not items:
        return base

    # Dedupe by URL (fallback to title)
    seen = set()
    deduped = []
    for h in items:
        key = (h.get("link") or "").strip().lower() or h.get("title", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    scored = []
    for h in deduped:
        s = ANALYZER.polarity_scores(h["title"])
        scored.append({**h, "compound": s["compound"]})

    pos = sum(1 for s in scored if s["compound"] >= 0.05)
    neg = sum(1 for s in scored if s["compound"] <= -0.05)
    neu = len(scored) - pos - neg
    avg = sum(s["compound"] for s in scored) / len(scored)

    by_source = {"news": 0, "hn": 0}
    by_locale = {}
    for s in scored:
        by_source[s["origin"]] = by_source.get(s["origin"], 0) + 1
        loc = s.get("locale", "en")
        by_locale[loc] = by_locale.get(loc, 0) + 1

    engagement = sum(s.get("engagement", 0) for s in scored)
    dates = [s["pubDate"] for s in scored if s.get("pubDate")]
    last_seen = max(dates) if dates else None

    by_strength = sorted(scored, key=lambda x: (-abs(x["compound"]), -(x.get("engagement") or 0)))
    sample = by_strength[:10]

    return {
        **base,
        "count": len(scored),
        "score": round(avg, 3),
        "pos": pos, "neu": neu, "neg": neg,
        "by_source": by_source,
        "by_locale": by_locale,
        "engagement": engagement,
        "last_seen": last_seen,
        "headlines": sample,
    }


def compute_all():
    companies = load_companies()
    buckets = {c["name"]: [] for c in companies}

    # Flatten all (company, task) pairs
    all_tasks = []
    for c in companies:
        for kind, args in build_news_tasks(c):
            all_tasks.append((c, kind, args))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_task, k, a, c): c for (c, k, a) in all_tasks}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                items = fut.result()
                if items:
                    buckets[c["name"]].extend(items)
            except Exception:
                pass

    results = [score_company(c, buckets[c["name"]]) for c in companies]
    results.sort(key=lambda r: (-r["score"], r["name"]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "summary": summarize(results),
        "companies": results,
    }


def summarize(results):
    covered = [r for r in results if r["count"] > 0]
    regions = {}
    for r in covered:
        regions.setdefault(r["region"], []).append(r["score"])
    region_avg = {k: round(sum(v) / len(v), 3) for k, v in regions.items()}

    pos = sum(1 for r in covered if r["score"] >= 0.05)
    neg = sum(1 for r in covered if r["score"] <= -0.05)
    neu = len(covered) - pos - neg

    source_totals = {"news": 0, "hn": 0}
    locale_totals = {}
    for r in covered:
        for k, v in r["by_source"].items():
            source_totals[k] = source_totals.get(k, 0) + v
        for k, v in r.get("by_locale", {}).items():
            locale_totals[k] = locale_totals.get(k, 0) + v

    return {
        "covered": len(covered),
        "no_coverage": len(results) - len(covered),
        "positive": pos, "neutral": neu, "negative": neg,
        "average_score": round(sum(r["score"] for r in covered) / len(covered), 3) if covered else 0.0,
        "region_average": region_avg,
        "source_totals": source_totals,
        "locale_totals": locale_totals,
        "total_mentions": sum(r["count"] for r in covered),
    }


# ------- History -------

def load_history():
    """Read all snapshots from data/history/YYYY-MM-DD.json, newest first."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    snapshots = []
    for path in sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")), reverse=True)[:60]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                snapshots.append(json.load(f))
        except Exception:
            continue
    return snapshots


# ------- Routes -------

@app.route("/api/sentiment")
def sentiment():
    payload = compute_all()
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "public, s-maxage=3600, stale-while-revalidate=86400"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/api/history")
def history():
    snaps = load_history()
    resp = make_response(jsonify({"snapshots": snaps, "count": len(snaps)}))
    resp.headers["Cache-Control"] = "public, s-maxage=3600, stale-while-revalidate=86400"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/")
def root():
    return send_from_directory(ROOT_DIR, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(ROOT_DIR, path)
