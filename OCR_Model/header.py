"""
Header discovery.

Turns the OCR'd header row into an ordered list of column names and the
correct top-to-bottom reading order for the data rows -- instead of
assuming a fixed, hardcoded schema.

Which grid row IS the header is no longer a static assumption passed in
from outside (structure.py's header_position flag, decided before any
OCR has even run). It's determined here, from the OCR'd content of the
grid's two extremal rows (top and bottom), using two independent
signals:

  1. Vocabulary match -- does this row's text match known column-name
     concepts (header_vocab.COLUMN_ALIASES)? Works for this table
     family and any other sheet whose columns are already listed there.
  2. Row-to-row similarity -- do the top and bottom rows read as
     near-duplicates of EACH OTHER, regardless of what they say? This
     is vocabulary-free: it catches a sheet that prints its header at
     both the top and the bottom (a common BOM/engineering-drawing
     layout) even for a completely unfamiliar table whose column names
     aren't in the vocabulary at all.

If a duplicate header is detected this way, BOTH copies are excluded
from the data rows -- only one supplies the column names, but neither
leaks through as a bogus data row. If neither signal is confident (an
unfamiliar table, badly garbled OCR on both ends, etc.), this falls
back to structure.py's original static assumption rather than guessing
wrong from weak evidence.

Stage 2 (ocr_paddle.py / ocr_targeted.py) already OCR's every cell in
the grid, including these rows -- this stage does not run any
additional OCR. It only reads what's already there and interprets it.
"""
from header_vocab import match_column_name, header_likeness_score, row_similarity


def _row_texts(struct, row_idx):
    return [struct.cells[(row_idx, c)].ocr_text for c in range(struct.n_cols)]


def _names_from_row(struct, row_idx):
    col_names = []
    seen = {}
    for c, raw in enumerate(_row_texts(struct, row_idx)):
        name = match_column_name(raw) or f"Col{c + 1}"
        # keep column names unique (e.g. two blank/unreadable header
        # cells shouldn't collapse into one column downstream)
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        col_names.append(name)
    return col_names


def _detect_header_row(struct, vocab_threshold=0.5, duplicate_threshold=0.6):
    """
    Returns (header_row, duplicate_row_or_None).

    Only ever considers the grid's two extremal rows (0 and n_rows-1)
    as header candidates -- a header, wherever it lives, sits at one
    end of the table, never in the middle.
    """
    top_row, bottom_row = 0, struct.n_rows - 1
    top_texts = _row_texts(struct, top_row)
    bottom_texts = _row_texts(struct, bottom_row)

    top_vocab = header_likeness_score(top_texts)
    bottom_vocab = header_likeness_score(bottom_texts)
    dup_score = row_similarity(top_texts, bottom_texts)

    if dup_score >= duplicate_threshold:
        # Both ends read as the same content -- a duplicated header.
        # Use whichever copy the vocabulary recognizes better (or the
        # bottom one, this table family's usual convention, as a
        # tie-break) to actually name the columns, and drop both from
        # the data.
        header_row = bottom_row if bottom_vocab >= top_vocab else top_row
        return header_row, (top_row if header_row == bottom_row else bottom_row)

    if top_vocab >= vocab_threshold or bottom_vocab >= vocab_threshold:
        header_row = top_row if top_vocab > bottom_vocab else bottom_row
        return header_row, None

    # Neither end reads confidently as a header (unfamiliar table,
    # garbled OCR on both ends, etc.) -- fall back to the original
    # static assumption rather than confidently picking a row that
    # isn't actually a header.
    return struct.header_row_idx, None


def build_header(struct):
    """
    Returns (col_names, data_row_order).

    col_names: one name per logical column (length == struct.n_cols),
    taken from the detected header row's own OCR text and normalized
    against the known column vocabulary. Falls back to the cleaned raw
    OCR text (or a positional "ColN" if that row's cell is blank) so a
    column the vocabulary doesn't recognize still gets a usable name --
    the number and identity of columns is discovered from the sheet,
    not assumed.

    data_row_order: struct row indices to treat as DATA, already in
    correct reading order, with the header row -- and any detected
    duplicate header row at the opposite end -- excluded. Because a
    header at the bottom of the image means the row directly above it
    is logically the first data row, that case reverses the image's
    own top-to-bottom row order.
    """
    header_row, duplicate_row = _detect_header_row(struct)
    col_names = _names_from_row(struct, header_row)

    exclude = {header_row}
    if duplicate_row is not None:
        exclude.add(duplicate_row)

    all_rows = [r for r in range(struct.n_rows) if r not in exclude]
    if header_row == struct.n_rows - 1:
        # header at the bottom -> image reads bottom-to-top
        data_row_order = sorted(all_rows, reverse=True)
    else:
        # header at the top -> image already reads top-to-bottom
        data_row_order = sorted(all_rows)

    return col_names, data_row_order
