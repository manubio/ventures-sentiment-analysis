"""Event category detection and story-level clustering.

When the same fundraise or layoff is reported by 3 outlets, we want
to count and display it once. The algorithm:

  1. Tag each headline with an event_category via regex (fundraise,
     acquisition, layoff, lawsuit, breach, launch, leadership,
     bankruptcy, or "other").
  2. Cluster headlines for the same company by (event_category,
     pubweek). The pubweek is the ISO week of publication, which
     tolerates multi-day news cycles around a single event.
  3. For "other" headlines, cluster only when title signatures
     (top-5 significant tokens, sorted) match within the same week.
  4. Pick a canonical representative per cluster: lowest tier wins
     first, then highest engagement, then most recent pubDate.

The canonical headline gets `is_cluster_head=True` and `cluster_size`.
Siblings carry `cluster_id` pointing back to the head. Stats and
rankings use cluster heads only; the detail modal can surface
siblings for context.
"""

import re
from collections import defaultdict
from datetime import datetime, timezone

CLUSTER_WINDOW_DAYS = 14
OTHER_CLUSTER_WINDOW_DAYS = 5
CROSS_CATEGORY_MERGE_JACCARD = 0.35
CROSS_CATEGORY_MERGE_WINDOW_DAYS = 14

_EVENT_PATTERNS = [
    ("bankruptcy",  re.compile(
        r"\b(bankruptcy|insolvency|chapter\s+11|liquidation|receivership|administração\s+judicial|recuperação\s+judicial|quiebra|falência)\b",
        re.IGNORECASE)),
    ("fraud",       re.compile(
        r"\b(fraud|fraude|scam|estafa|embezzl|bribery|corruption)\b",
        re.IGNORECASE)),
    ("layoff",      re.compile(
        r"\b(layoffs?|lays?\s+off|lay(?:\s+off|ing\s+off)|jobs?\s+cut|cuts?\s+(?:jobs|workforce|headcount|staff)|demissões?|demite|despidos?|recortes?\s+de\s+personal)\b",
        re.IGNORECASE)),
    ("lawsuit",     re.compile(
        r"\b(lawsuit|sued|suing|class[-\s]?action|files?\s+suit|legal\s+action|processa(?:d[oa])?|demanda(?:d[oa])?)\b",
        re.IGNORECASE)),
    ("breach",      re.compile(
        r"\b(security\s+breach|data\s+breach|hacked?|hack|leaked?|data\s+leak|vazamento|filtración)\b",
        re.IGNORECASE)),
    ("investigation", re.compile(
        r"\b(investigation|probe|regulatory\s+(?:probe|action)|investigation\s+into|investiga(?:ç|c)ão|investigación)\b",
        re.IGNORECASE)),
    ("leadership",  re.compile(
        r"\b(ceo|cto|cfo|coo|president|founder|fundador)\b.*\b(fired|ousted|resigns?|resigned|steps?\s+down|quits?|departs?|new\s+hire|appoint(?:ed|s)?|named|joins?|renuncia|deja|sai|nomead[oa])\b"
        r"|\b(appointed|named)\s+(?:new\s+)?(ceo|cto|cfo|coo|president)\b",
        re.IGNORECASE)),
    ("fundraise",   re.compile(
        r"\b(raises?|raised|raising|funding|secures?|secured|closed?\s+(?:a\s+)?(?:round|funding)"
        r"|series\s+[a-h]\b|\bipo\b|unicorn|unicórnio|valuation|valued\s+at"
        r"|captou?|capta|levanta(?:ou)?|recauda(?:d[oa]|ó)?|rodada|ronda\s+de"
        r"|boost(?:s|ed|ing)?\s+value|value\s+(?:above|of|at)\s+\$?\s*\d"
        r"|eyes?\s+(?:a\s+)?\$?\s*\d+(?:\.\d+)?\s*(?:m|mn|million|bn|billion)\s+(?:funding|round|raise)"
        r"|in\s+talks\s+to\s+(?:raise|secure|boost|close)"
        r"|nears?\s+(?:a\s+)?\$?\s*\d+.*(?:valuation|billion|unicorn)"
        r"|seeks?\s+(?:a\s+)?\$?\s*\d+.*(?:funding|round)"
        r"|funding\s+round\s+at\s+\$"
        r"|\$\d+[\d.,]*\s*(?:m|mn|million|bn|billion)\s+(?:at\s+)?(?:round|valuation|funding)"
        r")\b",
        re.IGNORECASE)),
    ("acquisition", re.compile(
        r"\b(acquires?|acquired|acquisition|takeover|merges?\s+with|merged\s+with|merger|adquire|adquirió|fusión|fusão)\b",
        re.IGNORECASE)),
    ("partnership", re.compile(
        r"\b(partners?\s+with|partnership|alliance|parceria|aliança|alianza)\b",
        re.IGNORECASE)),
    ("launch",      re.compile(
        r"\b(launches?|launched|unveils?|debuts?|rolls?\s+out|releases?|introduces?|lança(?:ou)?|lanzamiento|lanza)\b",
        re.IGNORECASE)),
    ("expansion",   re.compile(
        r"\b(expands?\s+to|enters?\s+(?:new\s+)?market|opens?\s+in|opens?\s+its\s+first|rollout\s+in|expande|expansión|expansão)\b",
        re.IGNORECASE)),
]


def event_category(title):
    """Return one of the known event labels or 'other'."""
    for label, rx in _EVENT_PATTERNS:
        if rx.search(title or ""):
            return label
    return "other"


_SIG_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{3,}")
_SIG_STOPS = set("the and for with from into over under about after before report reports says said news new via".split())


def _title_signature(title):
    """Top-5 significant tokens, sorted — a rough fingerprint for
    "same story" detection when no event pattern matches."""
    if not title:
        return ""
    toks = [t.lower() for t in _SIG_TOKEN_RE.findall(title)
            if t.lower() not in _SIG_STOPS]
    return " ".join(sorted(toks[:5]))


def _pubts(iso_or_rfc):
    """Parse a pubDate string to epoch seconds, or None."""
    if not iso_or_rfc:
        return None
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(iso_or_rfc, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _canonical_rank(h):
    """Sort key for picking the canonical headline in a cluster.

    Lowest tier first (1 < 2 < 3), then highest engagement, then most
    recent pubDate. Ties fall back to title length (shorter = cleaner).
    """
    tier = h.get("tier", 2)
    eng = h.get("engagement", 0) or 0
    ts = _pubts(h.get("pubDate")) or 0
    return (tier, -eng, -ts, len(h.get("title", "")))


def _greedy_chain(sorted_members, window_days):
    """Chain consecutive items into clusters as long as each next item
    falls within `window_days` of the PREVIOUS item (not of the first).
    Items without a pubDate get grouped into a single 'undated' cluster
    so one undated fundraise doesn't fragment into 10 singletons.
    """
    dated, undated = [], []
    for h in sorted_members:
        if _pubts(h.get("pubDate")):
            dated.append(h)
        else:
            undated.append(h)

    dated.sort(key=lambda h: _pubts(h.get("pubDate")) or 0)
    clusters = []
    current = []
    last_ts = None
    window_s = window_days * 86400
    for h in dated:
        ts = _pubts(h.get("pubDate"))
        if current and last_ts is not None and (ts - last_ts) > window_s:
            clusters.append(current)
            current = []
        current.append(h)
        last_ts = ts
    if current:
        clusters.append(current)
    if undated:
        clusters.append(undated)
    return clusters


def cluster_headlines(headlines):
    """Attach cluster_id / cluster_size / is_cluster_head to headlines
    and return (all_headlines, cluster_heads).

    Pipeline:
      1. Tag each headline with event_category.
      2. Partition by category.
      3. Within each non-'other' category, greedy-chain by pubDate
         within CLUSTER_WINDOW_DAYS.
      4. 'other' headlines partition by title signature first, then
         chain with a tighter window.
    """
    if not headlines:
        return [], []

    for h in headlines:
        h["event_category"] = event_category(h.get("title", ""))

    clusters_out = []
    by_cat = defaultdict(list)
    for h in headlines:
        by_cat[h["event_category"]].append(h)

    for cat, members in by_cat.items():
        if cat == "other":
            by_sig = defaultdict(list)
            for h in members:
                sig = _title_signature(h.get("title", ""))
                if sig:
                    by_sig[sig].append(h)
                else:
                    # headline with no meaningful tokens — own cluster
                    clusters_out.append([h])
            for sig, sig_members in by_sig.items():
                clusters_out.extend(_greedy_chain(sig_members, OTHER_CLUSTER_WINDOW_DAYS))
        else:
            clusters_out.extend(_greedy_chain(members, CLUSTER_WINDOW_DAYS))

    # Cross-category merge: if two clusters for the same company have
    # head titles with high token overlap and pubdates close together,
    # they're likely the same story tagged under different categories
    # ("raises $X" + "worth $X" talking about the same round).
    clusters_out = _merge_similar_clusters(clusters_out)

    # Proximity merge: for same-company strong-signal clusters whose
    # centers-of-mass are within 5 days, assume they're covering the
    # same underlying event regardless of token wording. This catches
    # "CuspAI raises $200m" + "Temasek-backed CuspAI set to boost
    # value above $1B" — same story, no shared content tokens.
    clusters_out = _merge_by_proximity(clusters_out)

    cluster_heads = []
    for idx, members in enumerate(clusters_out):
        members.sort(key=_canonical_rank)
        head = members[0]
        head["is_cluster_head"] = True
        head["cluster_id"] = idx
        head["cluster_size"] = len(members)
        head["cluster_sources"] = [
            {"source": m.get("source", ""), "link": m.get("link", ""),
             "tier": m.get("tier", 2), "pubDate": m.get("pubDate", ""),
             "title": m.get("title", "")}
            for m in members
        ]
        for sib in members[1:]:
            sib["is_cluster_head"] = False
            sib["cluster_id"] = idx
        cluster_heads.append(head)

    return headlines, cluster_heads


def _title_tokens(title):
    return set(
        t.lower() for t in _SIG_TOKEN_RE.findall(title or "")
        if t.lower() not in _SIG_STOPS and len(t) > 2
    )


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cluster_center_ts(cluster):
    ts = [_pubts(h.get("pubDate")) for h in cluster if _pubts(h.get("pubDate"))]
    return sum(ts) / len(ts) if ts else None


def _merge_by_proximity(clusters, window_days=5):
    """Merge clusters whose centers-of-mass are within `window_days`
    AND either share a non-trivial event category, OR both contain at
    least one strong-signal head (|compound| ≥ 0.3).

    Rationale: within a single company, multiple distinct big events
    within 5 days is vanishingly rare. When our token-Jaccard
    check fails (different outlets use different wording with no
    shared content tokens), proximity alone is a strong-enough
    dedup signal.
    """
    if len(clusters) <= 1:
        return clusters

    def canonical_of(c):
        return sorted(c, key=_canonical_rank)[0]

    window_s = window_days * 86400
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                a, b = clusters[i], clusters[j]
                ca = _cluster_center_ts(a)
                cb = _cluster_center_ts(b)
                if not (ca and cb) or abs(ca - cb) > window_s:
                    j += 1
                    continue
                ha = canonical_of(a)
                hb = canonical_of(b)
                # Same category (not "other") → merge
                same_cat = (ha.get("event_category") == hb.get("event_category")
                            and ha.get("event_category") not in (None, "other"))
                # Both strong signal → merge
                both_strong = (abs(ha.get("compound", 0)) >= 0.3
                               and abs(hb.get("compound", 0)) >= 0.3)
                if same_cat or both_strong:
                    a.extend(b)
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return clusters


def _merge_similar_clusters(clusters):
    """Greedy pairwise merge: if two clusters' canonical titles share
    ≥ CROSS_CATEGORY_MERGE_JACCARD of tokens and their average pubdates
    are within the window, merge them. Repeat until stable.
    """
    if len(clusters) <= 1:
        return clusters

    def canonical_of(cluster):
        s = sorted(cluster, key=_canonical_rank)
        return s[0]

    window_s = CROSS_CATEGORY_MERGE_WINDOW_DAYS * 86400
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                a, b = clusters[i], clusters[j]
                ta = _title_tokens(canonical_of(a)["title"])
                tb = _title_tokens(canonical_of(b)["title"])
                if _jaccard(ta, tb) < CROSS_CATEGORY_MERGE_JACCARD:
                    j += 1
                    continue
                ca = _cluster_center_ts(a)
                cb = _cluster_center_ts(b)
                if ca and cb and abs(ca - cb) > window_s:
                    j += 1
                    continue
                a.extend(b)
                clusters.pop(j)
                changed = True
            i += 1
    return clusters
