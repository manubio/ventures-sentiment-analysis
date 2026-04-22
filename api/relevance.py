"""Relevance filter: which headlines are actually about the target company?

Google News + HN return results whose text merely *contains* our query
string. That leaves plenty of false positives: "Honor" the home-care
company picks up articles about Huawei's Honor phone brand, "Rain" the
earned-wage-access startup picks up weather stories, etc.

Strategy: mark each headline with a relevance score in [0, 1]. A
headline must satisfy BOTH of:

  (a) Hard gate — the company name (or a clear variant/domain) appears
      as a whole-word token in the headline, OR the headline's link
      contains the company's domain.

  (b) Soft score — TF-IDF cosine between the headline and the company's
      descriptor document (description + region + explicit keywords).

A headline that fails (a) is dropped. Headlines that pass (a) are
accepted; the soft score is attached so the UI can de-emphasize low-
confidence matches. Structural patterns (funding announcements) are
exempted from (b) since those headlines are formulaic and rarely share
tokens with the company description.
"""

import math
import re
from collections import Counter

STOPWORDS = set("""
a about above after again against all am an and any are as at be because
been before being below between both but by could did do does doing down
during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just let me more
most my myself nor not now of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would
you your yours yourself yourselves
new says said news per via amid plus vs say
""".split())

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", re.UNICODE)

# "Strict" structural events: when a headline starts with the company
# name and matches one of these, we accept even without topic overlap.
# These are the events where "CompanyName <event>" is unambiguous
# (funding, M&A, failure, legal) — *not* operational news like
# "launches" or "expands" which a municipality can also do.
STRICT_EVENT_RX = re.compile(
    r"\b("
    r"raises?|raised|raising|secures?|secured|closed?\s+(?:a\s+)?(?:round|funding)"
    r"|series\s+[a-h]\b|\bipo\b|unicorn|valuation|valued\s+at"
    r"|boost(?:s|ed|ing)?\s+value|in\s+talks\s+to\s+(?:raise|secure|boost)"
    r"|eyes?\s+(?:a\s+)?\$|seeks?\s+(?:a\s+)?\$"
    r"|acquires?|acquired|acquisition|takeover|merges?\s+with|merger"
    r"|lawsuits?|sued|class[-\s]?action|settles?|settlement"
    r"|bankruptcy|insolvent|insolvency|chapter\s+11|files?\s+for"
    r"|(data|security)\s+breach|hacked|fraud|scandal"
    r"|layoffs?|lays?\s+off|lay(?:\s+off|ing\s+off)|mass\s+layoffs"
    r"|ceo\s+(?:resigns?|fired|ousted|steps?\s+down)"
    r"|capta|captou|levanta(?:ou)?|recauda(?:d[oa]|ó)?|rodada|ronda\s+de"
    r"|adquire|adquiere|despidos?|demissões?|quiebra|falência"
    r"|amplia\s+perdas|widens?\s+losses"
    r")\b",
    re.IGNORECASE,
)


def tokenize(text):
    if not text:
        return []
    return [
        t.lower() for t in _WORD_RE.findall(text)
        if len(t) > 2 and t.lower() not in STOPWORDS
    ]


class RelevanceScorer:
    """Builds IDF across the whole portfolio corpus once, then scores."""

    def __init__(self, companies):
        self.companies = companies
        # One TOPIC doc per company: description + region + keywords.
        # We deliberately exclude the company name and website so that
        # topic-overlap cosine reflects what the company DOES, not whether
        # a headline happens to contain the brand word. The `keywords`
        # field (in companies.json) enriches this with distinguishing
        # tokens like "chatbot WhatsApp" for Luzia so funding headlines
        # mentioning the sector still register overlap.
        self._topic_docs = {}
        for c in companies:
            topic_doc = " ".join([
                c.get("description", ""),
                c.get("region", ""),
                c.get("keywords", ""),
                c.get("type", "") or "",
            ])
            self._topic_docs[c["name"]] = topic_doc

        docs = list(self._topic_docs.values())
        N = max(1, len(docs))
        df = Counter()
        for doc in docs:
            for t in set(tokenize(doc)):
                df[t] += 1
        self._idf = {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._company_vecs = {
            name: self._vectorize(doc) for name, doc in self._topic_docs.items()
        }
        # Pre-compute name tokens (for whole-word matching)
        self._name_tokens = {
            c["name"]: self._name_variants(c)
            for c in companies
        }

    def _name_variants(self, company):
        """Tokens that count as a "name match" for this company."""
        name = company["name"]
        variants = {name}
        # Split on spaces & punctuation: "Captain Fresh" matches either
        for tok in re.split(r"[\s\-/.&]+", name):
            if len(tok) > 2:
                variants.add(tok)
        # Strip parentheticals: "Anecdote (Clarity)" → both
        m = re.match(r"^([^(]+?)\s*\(([^)]+)\)\s*$", name)
        if m:
            variants.add(m.group(1).strip())
            variants.add(m.group(2).strip())
        # Website domain root: "meesho.com" → "meesho"
        dom = (company.get("website") or "").split(".")[0]
        if dom and len(dom) > 2:
            variants.add(dom)
        return {v.lower() for v in variants if v}

    def _vectorize(self, text):
        tf = Counter(tokenize(text))
        if not tf:
            return {}
        total = sum(tf.values())
        return {t: (c / total) * self._idf.get(t, 1.0) for t, c in tf.items()}

    def _cosine(self, v1, v2):
        if not v1 or not v2:
            return 0.0
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[t] * v2[t] for t in common)
        n1 = math.sqrt(sum(x * x for x in v1.values()))
        n2 = math.sqrt(sum(x * x for x in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def score(self, company, headline):
        """Return (relevant_bool, confidence_float).

        Topic-first filter — a name match alone is NEVER enough:

          1. Hard gate — name variant in title OR domain in URL.
          2. Domain in URL → accept (strongest possible signal).
          3. Topic cosine ≥ 0.05 → accept (the headline is about the
             same subject area as the company). The company's
             topic vector is built from description + region + keywords
             + type; the company name itself is excluded.
          4. Strict-event fallback: name appears at position 0 AND the
             title matches STRICT_EVENT_RX (funding / M&A / layoffs /
             bankruptcy / lawsuit / breach / fraud). This catches
             formulaic headlines like "Meesho raises $50M" where the
             topic vector may not overlap with the funding-event
             phrasing, but position 0 + structural event is strong
             enough evidence.
          5. Otherwise drop — a name match without topic evidence is a
             likely homograph (LUZIA the restaurant, Santa Luzia the
             city, Honor the smartphone brand, Rain the weather, etc.).
        """
        title = headline.get("title", "") or ""
        link = (headline.get("link") or "").lower()

        title_tokens = tokenize(title)
        name_variants = self._name_tokens[company["name"]]
        name_positions = [i for i, tok in enumerate(title_tokens) if tok in name_variants]

        domain = (company.get("website") or "").lower()
        dom_hit = bool(domain) and domain in link

        # 1. Hard gate
        if not name_positions and not dom_hit:
            return (False, 0.0)

        # 2. Domain in URL — strongest signal, always accept
        if dom_hit:
            return (True, 0.85)

        # 3. Topic cosine
        tvec = self._vectorize(title)
        cos = self._cosine(tvec, self._company_vecs[company["name"]])
        if cos >= 0.05:
            return (True, cos)

        # 4. Strict-event fallback: name at position 0 + structural event
        if name_positions and name_positions[0] == 0 and STRICT_EVENT_RX.search(title):
            return (True, 0.40)

        # 5. Drop
        return (False, 0.0)
