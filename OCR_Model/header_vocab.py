"""
Header vocabulary.

Instead of hardcoding "this table has exactly these N columns in this
order", the pipeline now discovers however many columns the geometry
found (structure.py, unchanged) and whatever text OCR actually read out
of the header row, then only uses this vocabulary to *normalize* that
text -- e.g. mapping a noisy "OTY" or "QUANTITY" read back to the same
canonical "QTY" so postprocess.py's correctors still know which cleanup
rule applies.

If a header doesn't match anything in this vocabulary closely enough,
it is NOT forced into one of these buckets or dropped -- the cleaned
raw OCR text is kept as the column name as-is. That's what makes this
generalized: an unfamiliar sheet with different columns still produces
a usable table, just without a domain-specific corrector attached.
"""
import re
import difflib

# canonical column name -> known header phrasings / common OCR variants
# it can show up as. Extend this list as new sheet layouts are seen --
# it never needs to match the exact column count or order of any one
# document.
COLUMN_ALIASES = {
    "SL": ["SL", "SL NO", "SL. NO.", "S NO", "S.NO", "SR NO", "SERIAL",
           "SERIAL NO", "SI NO", "SLNO"],
    "PART DESCRIPTION": ["PART DESCRIPTION", "DESCRIPTION", "PART DESC",
                          "PARTS DESCRIPTION", "ITEM DESCRIPTION"],
    "QTY": ["QTY", "QTY.", "QUANTITY", "OTY"],
    "TML PART NO": ["TML PART NO", "TML PART NO.", "PART NO", "PART NO.",
                     "PART NUMBER"],
    "PIPE CUT LENGTH(mm)": ["PIPE CUT LENGTH", "PIPE CUT LENGTH(MM)",
                             "CUT LENGTH", "CUT LENGTH(MM)", "LENGTH(MM)",
                             "LENGTH"],
    "BUNCH & LOOSE": ["BUNCH & LOOSE", "BUNCH AND LOOSE", "BUNCH LOOSE",
                       "B & L", "B&L"],
    "COLOUR CODING": ["COLOUR CODING", "COLOR CODING", "COLOUR CODE",
                       "COLOR CODE"],
    "REFERENCE DRG NO.": ["REFERENCE DRG NO", "REFERENCE DRG NO.",
                           "REF DRG NO", "REFERENCE DRAWING NO", "DRG NO",
                           "DRAWING NO"],
}


def _clean(text):
    return re.sub(r'\s+', ' ', (text or '').strip().upper())


def match_column_name(raw_text, threshold=0.6):
    """
    Fuzzy-matches one OCR'd header cell's text against the known
    vocabulary of column concepts.

    Returns:
      - the canonical name if a close-enough alias match is found
      - otherwise the cleaned raw OCR text, so an unrecognized column
        still gets a usable (if uncorrected) name
      - None if there was no real text to work with at all (caller
        decides on a positional fallback like "Col3")
    """
    cleaned = _clean(raw_text)
    if not cleaned:
        return None

    best_name, best_ratio = None, 0.0
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            ratio = difflib.SequenceMatcher(None, cleaned, alias).ratio()
            if ratio > best_ratio:
                best_name, best_ratio = canonical, ratio

    if best_ratio >= threshold:
        return best_name
    return cleaned


def header_likeness_score(raw_texts, threshold=0.6):
    """
    Scores an entire row against the column vocabulary: the fraction of
    its cells whose cleaned text is a strong (>=threshold) match to some
    known column-name alias. A genuine header row is expected to have
    most of its cells match something in COLUMN_ALIASES; an ordinary
    data row (part descriptions, numbers, banners) generally won't. An
    empty cell counts as a non-match rather than being skipped, since a
    real header shouldn't have blank column labels.

    This only recognizes column concepts already listed above -- it's
    one signal among others callers should combine with a vocabulary-
    free check (e.g. comparing two candidate rows to each other) so an
    unfamiliar table's header can still be identified even when none of
    its column names are in this vocabulary yet.
    """
    if not raw_texts:
        return 0.0
    hits = 0
    for raw in raw_texts:
        cleaned = _clean(raw)
        if not cleaned:
            continue
        best_ratio = 0.0
        for aliases in COLUMN_ALIASES.values():
            for alias in aliases:
                ratio = difflib.SequenceMatcher(None, cleaned, alias).ratio()
                best_ratio = max(best_ratio, ratio)
        if best_ratio >= threshold:
            hits += 1
    return hits / len(raw_texts)


def row_similarity(texts_a, texts_b):
    """
    Vocabulary-free signal: how similar are two rows' texts to each
    other, column by column? Two rows that are near-duplicates of one
    another -- regardless of what they actually say, or whether any of
    it matches a known column name -- is strong evidence they're the
    same physical header line printed twice (e.g. once at the top and
    once at the bottom of a sheet). This lets header duplication be
    detected even on a table whose column vocabulary has never been
    seen before.
    """
    if not texts_a or not texts_b or len(texts_a) != len(texts_b):
        return 0.0
    ratios = []
    for a, b in zip(texts_a, texts_b):
        ca, cb = _clean(a), _clean(b)
        if not ca and not cb:
            continue  # both blank tells us nothing either way
        ratios.append(difflib.SequenceMatcher(None, ca, cb).ratio())
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)
