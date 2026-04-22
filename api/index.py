import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from flask import Flask, jsonify, make_response, send_from_directory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()
USER_AGENT = "Mozilla/5.0 (compatible; VenturesSentimentBot/1.0)"
MAX_HEADLINES = 15
FETCH_TIMEOUT = 8
MAX_WORKERS = 24

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
DATA_PATH = os.path.join(ROOT_DIR, "data", "companies.json")

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")


def load_companies():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["companies"]


def build_query(company):
    if company.get("search"):
        return company["search"]
    return f'"{company["name"]}"'


def fetch_headlines(company):
    query = urllib.parse.quote(build_query(company))
    url = f"https://news.google.com/rss/search?q={query}+when:90d&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            source = ""
            src_el = item.find("source")
            if src_el is not None and src_el.text:
                source = src_el.text.strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                items.append({"title": title, "source": source, "link": link, "pubDate": pub})
            if len(items) >= MAX_HEADLINES:
                break
        return items
    except Exception:
        return []


def score_company(company):
    headlines = fetch_headlines(company)
    base = {
        "name": company["name"],
        "website": company["website"],
        "region": company["region"],
        "description": company.get("description", ""),
        "count": 0,
        "score": 0.0,
        "pos": 0,
        "neu": 0,
        "neg": 0,
        "headlines": [],
    }
    if not headlines:
        return base

    scored = []
    for h in headlines:
        s = ANALYZER.polarity_scores(h["title"])
        scored.append({**h, "compound": s["compound"]})

    pos = sum(1 for s in scored if s["compound"] >= 0.05)
    neg = sum(1 for s in scored if s["compound"] <= -0.05)
    neu = len(scored) - pos - neg
    avg = sum(s["compound"] for s in scored) / len(scored)

    scored.sort(key=lambda x: x["compound"], reverse=True)
    top = scored[:3]
    bottom = [s for s in scored[-3:] if s not in top]

    return {
        **base,
        "count": len(scored),
        "score": round(avg, 3),
        "pos": pos,
        "neu": neu,
        "neg": neg,
        "headlines": top + bottom,
    }


def compute_all():
    companies = load_companies()
    results = [None] * len(companies)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(score_company, c): i for i, c in enumerate(companies)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                c = companies[i]
                results[i] = {
                    "name": c["name"],
                    "website": c["website"],
                    "region": c["region"],
                    "description": c.get("description", ""),
                    "count": 0, "score": 0.0, "pos": 0, "neu": 0, "neg": 0,
                    "headlines": [],
                }

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

    return {
        "covered": len(covered),
        "no_coverage": len(results) - len(covered),
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "average_score": round(sum(r["score"] for r in covered) / len(covered), 3) if covered else 0.0,
        "region_average": region_avg,
    }


@app.route("/api/sentiment")
def sentiment():
    payload = compute_all()
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "public, s-maxage=3600, stale-while-revalidate=86400"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/")
def root():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(PUBLIC_DIR, path)
