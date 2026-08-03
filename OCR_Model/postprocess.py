"""
Post-processing / correction layer.

No domain word vocabulary here (no lists of known part-description
words like "PIPE"/"ASSY", no known-colour lists, etc.) -- correction is
purely structural/typographic: fixing characters that are visually
confusable with digits, and only where the cell already looks numeric.
A cell that isn't mostly digits is left exactly as OCR read it, since
"is this a valid word" can't be judged without a word list, and this
pipeline intentionally doesn't hardcode one.
"""
import re

# Common OCR digit/letter confusions seen in tesseract output on this
# font, based on real samples captured from the pipeline's own output.
# This is a character-level typographic table (which glyphs look like
# which), not a word/domain vocabulary -- it applies the same regardless
# of what column or document this is.
DIGIT_CONFUSION = {
    'O': '0', 'o': '0', 'Q': '0',
    'l': '1', 'I': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2',
    'G': '6',
    'T': '7',
    '一': '1',  # PaddleOCR-specific: CJK "one" glyph occasionally emitted
                # in place of the digit 1. Unlike the other entries above,
                # this is never a legitimate reading anywhere in this
                # document (there's no scenario where a real "一" belongs
                # in a BOM), so it's safe as a global, unconditional
                # mapping -- no column-context needed.
}


# Characters that are safe to treat as "digit-like" for gate purposes
# even in their raw, unmapped form -- unlike O/I/S/etc. (which are
# ordinary English letters that legitimately appear inside real words
# like "DRAWING", so counting them here would misjudge free text as
# numeric), these can never legitimately appear in this document's
# English free text at all. So treating them as digit-like doesn't
# reintroduce the false-positive risk the gate exists to avoid.
_UNAMBIGUOUS_DIGIT_LOOKALIKES = {'一'}


def _normalize_if_numeric(text, min_digit_fraction=0.5):
    """
    Generic structural cleanup: if a cell is already mostly digits
    (after fixing common OCR letter/digit look-alikes), collapse it down
    to its longest digit run. If it's mostly letters/words, it's left
    completely untouched -- correcting free text requires knowing what
    the "right" word should be, which needs a word vocabulary, and this
    pipeline doesn't assume one.
    """
    if not text or not text.strip():
        return text

    # Classification gate uses the RAW text's own digit fraction -- letters
    # are not pre-mapped to digits here, or a free-text banner like
    # "REFER DRAWING NO.517443900141" gets misjudged as numeric just
    # because "DRAWING"/"NO" happen to contain I/G/O look-alikes. The one
    # exception is _UNAMBIGUOUS_DIGIT_LOOKALIKES (see above) -- those are
    # safe to count here precisely because they can't appear in real words.
    alnum_count = sum(ch.isalnum() for ch in text)
    raw_digit_count = sum(
        ch.isdigit() or ch in _UNAMBIGUOUS_DIGIT_LOOKALIKES for ch in text
    )
    if alnum_count == 0 or (raw_digit_count / alnum_count) < min_digit_fraction:
        return text  # not numeric-looking -- leave as-is, don't guess

    # Only once a cell has already qualified as numeric do we apply the
    # confusion table, to clean up misread digits within it.
    corrected = ''.join(DIGIT_CONFUSION.get(ch, ch) for ch in text)

    runs = re.findall(r'\d+', corrected)
    if not runs:
        return text
    return max(runs, key=len)


def correct_cell(column_name, text):
    """
    Applies the pipeline's corrections. Two tiers, kept deliberately
    separate:

    1. The generic, vocabulary-free numeric cleanup (_normalize_if_numeric)
       -- applies everywhere, regardless of column, as before.

    2. A narrow, column-SCOPED closed-vocabulary corrector, used only for
       columns whose entire valid answer space is a tiny fixed set (right
       now: BUNCH & LOOSE, which is only ever "B" or "L"). This is a
       deliberate, limited exception to this file's general "no domain
       vocabulary" policy -- justified specifically because the answer
       space here is small and fully enumerable, so snapping to the
       nearest valid option can't introduce a wrong-but-plausible answer
       the way it could for open-ended text (a part description, a colour
       name typo, etc). Do not extend this pattern to columns without a
       similarly tiny, exhaustively-known set of valid values.
    """
    if column_name == "BUNCH & LOOSE":
        return _snap_to_closed_vocab(text, {"B", "L"})
    return _normalize_if_numeric(text)


def _snap_to_closed_vocab(text, valid_values):
    """Column-scoped only -- see correct_cell docstring. Snaps a cell's
    text to the closest single valid value if it's a plausible near-miss
    (single alnum character not already in the valid set), otherwise
    leaves it untouched rather than guessing."""
    if not text or not text.strip():
        return text
    cleaned = text.strip().upper()
    if cleaned in valid_values:
        return cleaned
    if len(cleaned) == 1 and cleaned.isalnum():
        # Known PaddleOCR confusions specific to this closed set. '8' is
        # the empirically observed misread of 'B' in this font/model;
        # extend this mapping if other confusions turn up in practice.
        SINGLE_CHAR_FIXES = {'8': 'B'}
        return SINGLE_CHAR_FIXES.get(cleaned, cleaned)
    return text
