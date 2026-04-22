"""Pattern-based compound-score overrides applied BEFORE VADER.

VADER + lexicon is bag-of-words; it has no grammar. So "fails to raise
$10M" scores positive (it sees "raise"/"funding") and "avoids bankruptcy"
scores negative. These regex rules catch the most common business
patterns where grammar flips the sign, plus strong structural signals
(files for bankruptcy, series C raise, goes public) where we want a
high-magnitude override.

Each rule: (regex, score). If multiple rules match a headline, we take
the one with the largest absolute score (strongest signal wins).
"""

import re

_FLAGS = re.IGNORECASE | re.UNICODE

# Order is informational only; selection is by max |score|.
PATTERNS_EN = [
    # --- Negated positive events: a GOOD thing that didn't happen (bad)
    (r'\b(fails?|failed|failing|unable)\s+to\s+(close|raise|secure|land|get|attract|complete|finalize|reach|win|acquire|merge|launch|ipo|list)', -0.75),
    (r'\bcannot\s+(close|raise|secure|acquire|launch)', -0.70),
    (r'\b(denied|rejected|turned\s+down|blocked|spurned)\s+\w*\s*(funding|merger|deal|acquisition|bid|offer|takeover)', -0.65),
    (r'\b(falls?\s+through|collapses?|breaks?\s+down|scrapped|shelved|abandoned|walks?\s+away\s+from)\b.*\b(deal|round|merger|acquisition|ipo|talks)', -0.80),
    (r'\bwithdraws?\s+(ipo|filing|offer|bid)', -0.65),
    (r'\b(loses?|losing|lost)\s+(\w+\s+){0,3}(deal|contract|customer|investor|partnership|client)', -0.55),
    (r'\bstalls?\b.*\b(round|funding|deal|ipo)', -0.55),

    # --- Negated negative events: a BAD thing that didn't happen (mild good)
    (r'\b(avoids?|averts?|averted|dodges?|fends?\s+off|wards?\s+off|survives?|survived)\s+(\w+\s+){0,3}(bankruptcy|collapse|lawsuit|suit|probe|layoffs|crisis|shutdown|insolvency|default|meltdown)', +0.50),
    (r'\b(denies?|dismiss(es|ed)?|rejects?|refutes?|disputes?)\s+(\w+\s+){0,3}(allegations?|claims?|rumors?|reports?|charges?|accusations?|wrongdoing)', +0.25),
    (r'\b(cleared|exonerated|acquitted|vindicated)\b', +0.55),
    (r'\bno\s+(layoffs?|breach|outage|bankruptcy|fraud|wrongdoing|misconduct)\b', 0.0),
    (r'\bwithout\s+(layoffs?|breach|outage|wrongdoing)\b', 0.0),

    # --- Rumor / speculative dampener (negative framing without confirmation)
    (r'\b(reportedly|allegedly|rumored|speculation|report\s+of|reports?\s+say)\b.*\b(layoffs?|bankruptcy|investigation|probe|lawsuit|breach|fraud)', -0.35),
    (r'\b(could|may|might|expected\s+to|likely\s+to)\s+(file|face|lose|cut|lay\s+off|shut|close)', -0.30),

    # --- Strong positive (structural)
    (r'\braises?\s+\$?\s*\d+(\.\d+)?\s*(m|mn|mm|million|bn|b|billion|k|thousand)\b', +0.80),
    (r'\b(raised|secured|closed|banked|landed|grabbed)\s+\$\s*\d', +0.75),
    (r'\bseries\s+[a-h]\b.*\b(round|raises?|closed?|secured?|funding)', +0.75),
    (r'\b(funding|round)\s+of\s+\$\s*\d', +0.70),
    (r'\b(valued|valuation)\s+(at|of)\s+\$\s*\d+\s*(m|million|bn|b|billion)', +0.65),
    (r'\b(goes?\s+public|public\s+listing|stock\s+market\s+debut|nasdaq|nyse)\s+(debut|listing|launch)', +0.70),
    (r'\bipo\s+(pricing|priced|debut|launches?|surges?|raises?)', +0.70),
    (r'\bbecomes?\s+(a\s+)?unicorn\b', +0.95),
    (r'\bhits?\s+\$?\s*\d+\s*(m|million|bn|billion)\s+(in\s+)?(revenue|arr|valuation|users)', +0.60),
    (r'\bacquires?\s+[A-Z][A-Za-z]+', +0.55),
    (r'\bto\s+acquire\b', +0.50),
    (r'\b(expands?\s+to|launches?\s+in|enters?)\s+(new\s+)?(market|country|region)', +0.40),
    (r'\bpartners?\s+with\s+[A-Z]', +0.40),
    (r'\b(wins?|won|awarded)\s+(\w+\s+){0,3}(contract|deal|award|recognition)', +0.55),
    (r'\bprofitable\s+(for\s+the\s+first|this\s+quarter|first\s+time)', +0.80),
    (r'\bbreaks?\s+even\b', +0.60),

    # --- Strong negative (structural)
    (r'\blays?\s+off\s+\d+(\.\d+)?\s*(%|percent)', -0.90),
    (r'\blays?\s+off\s+(\w+\s+){0,2}\d+\s+(staff|employees|workers|people)', -0.85),
    (r'\b(mass|massive|sweeping|major)\s+layoffs?\b', -0.90),
    (r'\blays?\s+off\b', -0.70),
    (r'\bfiles?\s+for\s+(bankruptcy|chapter\s+11|insolvency|administration)', -0.95),
    (r'\benters?\s+(administration|receivership|liquidation)', -0.90),
    (r'\b(shuts?\s+down|shutting\s+down|closes?\s+down|winding\s+down|winds?\s+down)\b', -0.75),
    (r'\bhit\s+with\s+(a\s+)?(lawsuit|class\s*action|fine|investigation|penalty)', -0.75),
    (r'\b(sued|sues|being\s+sued)\b', -0.55),
    (r'\b(class[-\s]?action|class\s+suit)\b', -0.70),
    (r'\b(security|data)\s+breach\b', -0.80),
    (r'\b(got\s+)?hacked\b', -0.75),
    (r'\b(fraud|scam)\s+(charges?|allegations?|investigation|case|scheme)', -0.90),
    (r'\b(accused|charged|indicted)\s+(of|with)\s+(fraud|wrongdoing|misconduct|embezzlement|bribery)', -0.90),
    (r'\b(ceo|cto|cfo|coo|founder)\s+(fired|ousted|resigns?|steps?\s+down|quits|exits|forced\s+out)', -0.55),
    (r'\bresignation\s+of\s+(ceo|cto|cfo|coo|founder)', -0.55),
    (r'\b(regulatory|regulator)\s+(probe|investigation|action|crackdown)', -0.70),
    (r'\bregulatory\s+(fine|penalty)', -0.70),
    (r'\b(stock|shares?)\s+(plunges?|plummets?|tumbles?|crashes?)', -0.80),
    (r'\bdowngraded\s+by\b', -0.60),
    (r'\b(missed|fell\s+short\s+of)\s+(earnings|expectations|guidance|targets)', -0.55),
    (r'\b(cuts?|slashing|slashes?)\s+(jobs|workforce|headcount|staff)', -0.80),
    (r'\b(delays?|postpones?|pushed\s+back)\s+(ipo|launch|rollout)', -0.50),
    (r'\bwarns?\s+of\s+(losses|bankruptcy|layoffs|slowdown)', -0.70),
    (r'\bunder\s+(investigation|probe|scrutiny)', -0.60),
]

# Portuguese & Spanish patterns (lighter — just the high-impact flips)
PATTERNS_PT = [
    # negated positive (didn't raise, couldn't close)
    (r'\b(não\s+consegue|fracassa|falhou|falhar)\s+em\s+(levantar|captar|fechar)', -0.75),
    (r'\b(rodada\s+)?frustrada\b', -0.70),
    (r'\bdesiste\s+da\s+(rodada|oferta|aquisição)', -0.65),
    # negated negative (avoided bankruptcy, denies layoffs)
    (r'\b(evita|evitou|escapa|escapou)\s+(da\s+)?(falência|demissão|demissões|crise)', +0.50),
    (r'\b(nega|desmente|rejeita)\s+(\w+\s+){0,3}(demissão|demissões|boato|rumor|falência)', +0.25),
    # strong positive patterns
    (r'\b(capta|captou|levanta|levantou|recebe|recebeu)\s+(us\$|r\$)?\s*\d+\s*(mi|milhões|bi|bilhões)', +0.80),
    (r'\bsérie\s+[a-f]\b.*\b(capta|rodada|investimento)', +0.75),
    (r'\bvira\s+unicórnio\b', +0.95),
    (r'\badquire\s+[A-Z]', +0.55),
    # strong negative
    (r'\b(demite|demitiu)\s+\d+', -0.85),
    (r'\bdemissão\s+em\s+massa', -0.90),
    (r'\b(pede|entra\s+em|decreta)\s+(falência|recuperação\s+judicial)', -0.95),
    (r'\bvazamento\s+de\s+dados', -0.80),
    (r'\bfraude\s+(de|em|acusado)', -0.90),
    (r'\b(ceo|fundador|cto)\s+(é\s+demitido|renuncia|deixa|sai)', -0.55),
]

PATTERNS_ES = [
    (r'\b(no\s+logra|fracasa|falló)\s+en\s+(levantar|recaudar|cerrar)', -0.75),
    (r'\b(abandona|desiste\s+de)\s+(la\s+)?(ronda|oferta|adquisición)', -0.65),
    (r'\b(evita|evitó|esquiva)\s+(la\s+)?(quiebra|despidos?|crisis)', +0.50),
    (r'\b(niega|desmiente|rechaza)\s+(\w+\s+){0,3}(despidos?|rumores?|quiebra)', +0.25),
    (r'\b(recauda|recaudó|levanta|levantó|capta|captó)\s+(us\$|\$)?\s*\d+\s*(m|millones|mm|mil\s+millones)', +0.80),
    (r'\bserie\s+[a-f]\b.*\b(ronda|inversión|capta|recauda)', +0.75),
    (r'\bse\s+convierte\s+en\s+unicornio\b', +0.95),
    (r'\badquiere\s+[A-Z]', +0.55),
    (r'\b(despide|despidió)\s+\d+', -0.85),
    (r'\bdespidos?\s+masivos?\b', -0.90),
    (r'\b(se\s+declara\s+en|entra\s+en)\s+quiebra\b', -0.95),
    (r'\bfiltración\s+de\s+datos', -0.80),
    (r'\bfraude\s+(de|en|acusado)', -0.90),
    (r'\b(ceo|fundador)\s+(despedido|renuncia|se\s+va)', -0.55),
]

_COMPILED = [
    (re.compile(p, _FLAGS), s) for p, s in (PATTERNS_EN + PATTERNS_PT + PATTERNS_ES)
]


def override_score(text):
    """Return a float override in [-1, +1] if any pattern matches, else None.

    When multiple rules match, the one with the largest |score| wins.
    """
    best = None
    for rx, score in _COMPILED:
        if rx.search(text):
            if best is None or abs(score) > abs(best):
                best = score
    return best
