"""Business/finance/startup lexicon extensions for VADER.

VADER's default lexicon is trained on social media and is near-blind to
business events: it scores "raises $18M" at 0.0, misses "layoffs",
"bankruptcy", "unicorn", etc. These additions capture the vocabulary
of funding rounds, M&A, regulatory actions, and operational setbacks.

Scores follow VADER's convention (-4 to +4). Values are deliberate:
- Neutral-ish events (partnership, launches):       +1.0 to +1.5
- Strong positive (raises, profits, unicorn):       +2.0 to +3.0
- Severe negative (bankruptcy, fraud, shutdown):    -3.0 to -3.8
"""

BUSINESS_LEXICON_EN = {
    # --- Funding / capital events (positive)
    "raises": 2.5, "raised": 2.5, "raising": 2.0,
    "funding": 1.5, "funded": 2.0, "funds": 1.0,
    "backed": 1.5, "backs": 1.5,
    "invests": 1.5, "invested": 1.5, "investment": 1.2,
    "ipo": 2.5, "listing": 1.5, "listed": 1.0,
    "unicorn": 3.0, "decacorn": 3.5,
    "valuation": 1.0, "valued": 1.0,
    "closes": 1.5, "closed": 1.5,  # context: "closes round"
    "oversubscribed": 3.0,
    "secures": 1.8, "secured": 1.8,

    # --- M&A / partnerships (positive)
    "acquires": 1.8, "acquired": 1.5, "acquisition": 1.5,
    "merger": 1.5, "merges": 1.5, "merged": 1.5,
    "partnership": 1.8, "partners": 1.2, "partnered": 1.5,
    "alliance": 1.8,
    "joint venture": 1.5,

    # --- Growth / operations (positive)
    "launches": 1.8, "launched": 1.8, "launch": 1.5,
    "expands": 1.8, "expansion": 1.8, "expanding": 1.8,
    "rollout": 1.5, "rolls out": 1.5,
    "scales": 1.5, "scaling": 1.5,
    "growth": 2.0, "growing": 1.5, "grows": 1.5, "grew": 1.5,
    "surge": 2.5, "surges": 2.5, "surging": 2.5,
    "soars": 2.8, "soaring": 2.5,
    "jumps": 1.8, "jumped": 1.8,
    "boost": 2.0, "boosts": 2.0, "boosted": 1.8,
    "accelerates": 2.0, "accelerating": 1.8,
    "breakthrough": 2.8,
    "milestone": 1.8,
    "record": 1.5,
    "profitable": 2.8, "profits": 2.0, "profit": 1.8, "profitability": 2.0,
    "revenue": 0.8,  # mild positive marker
    "doubles": 1.8, "triples": 2.0, "tripled": 2.0, "doubled": 1.8,
    "beats": 2.0, "beating": 1.5, "outperforms": 2.0,
    "wins": 2.0, "won": 1.5, "winning": 1.5,
    "approves": 1.5, "approved": 1.5, "approval": 1.5,
    "greenlight": 1.5, "greenlit": 1.5,
    "innovation": 1.5, "innovates": 1.5, "innovative": 1.5,
    "leads": 1.0, "leading": 1.2, "leader": 1.2,
    "dominant": 1.5, "dominate": 1.5, "dominates": 1.5,
    "hit": 1.0, "hits": 1.0,  # "hits 1M users"

    # --- Regulatory / legal (negative)
    "lawsuit": -2.8, "lawsuits": -2.8, "sued": -2.5, "suing": -2.5, "sues": -2.5,
    "investigation": -2.2, "investigating": -2.0, "probe": -2.2, "probed": -2.2,
    "subpoena": -2.5, "subpoenaed": -2.5,
    "fine": -1.8, "fined": -2.0, "penalty": -1.8, "penalties": -1.8,
    "sanctions": -2.5, "sanctioned": -2.5,
    "banned": -2.5, "bans": -2.2, "ban": -2.0,
    "blocked": -2.0, "blocks": -1.8,
    "raided": -2.5, "raid": -2.0,
    "compliance": -0.3,  # mild negative marker (usually appears with issues)
    "violation": -2.5, "violations": -2.5,
    "allegations": -2.5, "alleged": -1.8,

    # --- Failure / distress (negative)
    "layoffs": -3.0, "layoff": -3.0,
    "lays off": -3.0, "laid off": -3.0, "laying off": -3.0,
    "fires": -2.5, "firing": -2.5,  # "fires CEO"
    "shutdown": -3.0, "shutting down": -3.0, "shuts down": -3.0,
    "closing": -1.5, "closed down": -2.5,
    "bankruptcy": -3.8, "bankrupt": -3.5, "insolvent": -3.5, "insolvency": -3.5,
    "liquidation": -3.5, "winding down": -3.0, "wound down": -3.0,
    "default": -2.5, "defaults": -2.5, "defaulted": -2.5,
    "crisis": -2.8, "crises": -2.8,
    "struggle": -2.0, "struggling": -2.2, "struggles": -2.0,
    "collapse": -3.5, "collapses": -3.5, "collapsed": -3.5,
    "crashes": -3.0, "crashed": -3.0, "crash": -2.5,
    "plummets": -3.2, "plummeted": -3.2,
    "tumbles": -2.8, "tumbled": -2.8,
    "plunges": -2.8, "plunged": -2.8,
    "slumps": -2.5, "slump": -2.5, "slumped": -2.5,
    "fell": -1.2, "falls": -1.2,
    "declines": -1.8, "declined": -1.5, "declining": -1.8, "decline": -1.8,
    "loses": -1.8, "losses": -2.0, "loss": -1.8, "losing": -1.8,
    "misses": -1.8, "missed": -1.8, "miss": -1.5,
    "downgrade": -2.5, "downgraded": -2.5, "downgrades": -2.5,
    "cuts": -1.5, "cutting": -1.5,
    "recall": -2.5, "recalled": -2.5, "recalls": -2.5,
    "halt": -2.0, "halted": -2.2, "halts": -2.0,
    "delay": -1.5, "delayed": -1.5, "delays": -1.5,
    "scrap": -2.0, "scraps": -2.0, "scrapped": -2.0,
    "abandon": -2.2, "abandons": -2.2, "abandoned": -2.0,
    "weak": -1.8, "weakness": -1.8, "weakens": -1.8,

    # --- Security / reputational (negative)
    "breach": -2.5, "breached": -2.5, "breaches": -2.5,
    "hack": -2.5, "hacked": -2.8, "hackers": -2.2,
    "leaked": -2.5, "leak": -2.0,
    "exposed": -1.8, "exposes": -1.8,
    "fraud": -3.5, "fraudulent": -3.5,
    "scam": -3.5, "scammed": -3.5, "scams": -3.0,
    "scandal": -3.2, "scandals": -3.0,
    "controversy": -2.2, "controversial": -1.8,
    "criticized": -2.0, "criticism": -1.8,
    "backlash": -2.5,
    "threatens": -2.0, "threat": -1.8, "threats": -1.8,
    "concerns": -1.2, "concerned": -1.2,
    "risky": -1.5, "risks": -1.0,

    # --- Leadership change (mildly negative, ambiguous)
    "resigns": -1.8, "resignation": -1.8, "resigned": -1.8,
    "stepping down": -1.8, "steps down": -1.8, "stepped down": -1.8,
    "ousted": -2.8, "fires ceo": -3.0,
    "departure": -1.0, "departs": -1.0,

    # --- Recovery / improvement (positive)
    "recovery": 1.8, "recovers": 1.8, "rebounds": 2.0,
    "turnaround": 2.0,
    "improves": 1.5, "improvement": 1.5, "improved": 1.5,
    "stabilizes": 1.2, "stabilizing": 1.2,

    # --- Ambiguous — only include if clearly directional
    "exit": 0.5, "exits": 0.5,  # usually "good exit" in VC context
}

# --- Portuguese finance/startup lexicon (smaller, focused)
BUSINESS_LEXICON_PT = {
    # positive
    "capta": 2.5, "captou": 2.5, "captação": 2.0,
    "levanta": 2.5, "levantou": 2.5,  # "levanta rodada"
    "rodada": 1.0,
    "investe": 1.5, "investiu": 1.5, "investimento": 1.2,
    "cresce": 1.8, "cresceu": 1.8, "crescimento": 1.8,
    "lucro": 2.0, "lucra": 2.0, "lucrativa": 2.5, "lucrativo": 2.5,
    "parceria": 1.8,
    "expande": 1.8, "expansão": 1.8,
    "lança": 1.5, "lançou": 1.5, "lançamento": 1.5,
    "adquire": 1.8, "adquiriu": 1.8, "aquisição": 1.5,
    "avança": 1.5, "avançou": 1.5,
    "dispara": 2.5, "disparou": 2.5,
    "aumenta": 1.5, "aumentou": 1.5, "aumento": 1.5,
    "fusão": 1.5,
    # negative
    "prejuízo": -2.5, "prejuízos": -2.5,
    "queda": -1.8, "cai": -1.8, "caiu": -1.8,
    "quebra": -3.0, "quebrou": -3.0, "falência": -3.5, "falido": -3.5,
    "demite": -2.5, "demitiu": -2.5, "demissões": -3.0, "demissão": -2.5,
    "fechamento": -2.0, "fecha": -1.5, "fechou": -1.5,
    "processo": -1.2, "processada": -2.0, "processado": -2.0,
    "multa": -1.8, "multada": -1.8, "multado": -1.8,
    "fraude": -3.5, "escândalo": -3.0,
    "investigação": -2.2, "investiga": -2.0,
    "vazamento": -2.5,  # data leak
    "golpe": -3.0,
    "crise": -2.5,
    "encerra": -2.0, "encerrou": -2.0, "encerramento": -2.0,
    "dificuldades": -2.0, "dificuldade": -1.8,
    "polêmica": -2.0, "polêmico": -1.8,
    "crítica": -1.5, "criticada": -1.8, "criticado": -1.8,
    "atraso": -1.5, "atrasa": -1.5, "atrasou": -1.5,
    "desiste": -1.8, "desistiu": -1.8,
}

# --- Spanish finance/startup lexicon
BUSINESS_LEXICON_ES = {
    # positive
    "recauda": 2.5, "recaudó": 2.5, "recaudación": 2.0,
    "levanta": 2.5, "levantó": 2.5,
    "ronda": 1.0,
    "invierte": 1.5, "invirtió": 1.5, "inversión": 1.2,
    "crece": 1.8, "creció": 1.8, "crecimiento": 1.8,
    "ganancia": 2.0, "ganancias": 2.0, "rentable": 2.5,
    "alianza": 1.8, "asociación": 1.5,
    "expande": 1.8, "expansión": 1.8,
    "lanza": 1.5, "lanzó": 1.5, "lanzamiento": 1.5,
    "adquiere": 1.8, "adquirió": 1.8, "adquisición": 1.5,
    "avanza": 1.5,
    "dispara": 2.5, "disparó": 2.5,
    "aumenta": 1.5, "aumentó": 1.5, "aumento": 1.5,
    "fusión": 1.5,
    "supera": 1.5, "superó": 1.5,
    # negative
    "pérdida": -2.0, "pérdidas": -2.5,
    "caída": -1.8, "cae": -1.8, "cayó": -1.8,
    "quiebra": -3.5, "quebró": -3.5,
    "despidos": -3.0, "despido": -2.5, "despide": -2.5, "despidió": -2.5,
    "cierra": -1.5, "cerró": -1.5, "cierre": -2.0,
    "demanda": -1.8, "demandada": -2.0, "demandado": -2.0,
    "multa": -1.8, "multada": -1.8,
    "fraude": -3.5, "escándalo": -3.0,
    "investigación": -2.2, "investiga": -2.0,
    "filtración": -2.5,
    "estafa": -3.0,
    "crisis": -2.5,
    "dificultades": -2.0,
    "polémica": -2.0,
    "critica": -1.5, "criticada": -1.8, "criticado": -1.8,
    "retraso": -1.5, "retrasa": -1.5,
    "abandona": -2.2, "abandonó": -2.2,
}


def install(analyzer):
    """Patch the given VADER analyzer with business + multilingual terms."""
    for k, v in BUSINESS_LEXICON_EN.items():
        analyzer.lexicon[k.lower()] = v
    for k, v in BUSINESS_LEXICON_PT.items():
        analyzer.lexicon[k.lower()] = v
    for k, v in BUSINESS_LEXICON_ES.items():
        analyzer.lexicon[k.lower()] = v
