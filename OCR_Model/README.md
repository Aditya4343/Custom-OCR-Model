# BOM Table Extraction Pipeline

Three independent stages, plus a header-discovery step. Structure
(rows/columns/merges) is learned before any OCR runs, and OCR never
determines structure. The column *names* are no longer hardcoded either
-- they're read out of the sheet's own header row and normalized
against a small vocabulary, so the pipeline isn't locked to one fixed
8-column schema.

**These BOM sheets read bottom-up.** The title block sits at the bottom
of the page and the table is built upward from it, so the header row is
the BOTTOM row of the grid, not the top -- and the row directly above
the header is logically item 1, not the row at the top of the image.
The pipeline starts there: it reads the header row first to learn how
many columns exist and what they're called, builds the column structure
from that, then OCRs the rest of the table in correct (bottom-up)
reading order. Pass `--header=top` if a given sheet is oriented the
normal way instead.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

This installs PaddleOCR (the default OCR engine) and its `paddlepaddle`
inference backend, plus Tesseract's Python bindings as an optional
fallback engine. Tesseract also needs the system binary if you want to
use `--engine=tesseract`:

```bash
sudo apt-get install -y tesseract-ocr
```

PaddleOCR needs no external server or system package -- everything it
needs ships in the `paddleocr` / `paddlepaddle` pip packages, and model
weights download automatically on first use.

## Files
- `structure.py` — Stage 1: geometry only, no OCR. Also records which
  grid row is the header (`header_row_idx`, default: bottom row)
- `ocr_targeted.py` — Stage 2 (Tesseract engine): per-block OCR + same-value-vs-distinct decision
- `ocr_paddle.py` — Stage 2 (PaddleOCR engine, default): same block traversal/decision logic, PaddleOCR does the leaf read
- `header_vocab.py` — vocabulary of known column-header phrasings, used to normalize noisy OCR into a canonical column name
- `header.py` — Stage 2.5: turns the OCR'd header row into `(col_names, data_row_order)` — no fixed column list, no assumed row order
- `ocr_mapping.py` — diagnostics (confusion-matrix classification)
- `export_xlsx.py` — Stage 3: Excel export, in the row order `header.py` determined
- `main.py` — entry point tying all stages together

```bash
python3 main.py <input_image_path> [output_dir]                          # PaddleOCR, header at bottom (defaults)
python3 main.py <input_image_path> [output_dir] --engine=tesseract
python3 main.py <input_image_path> [output_dir] --header=top             # sheet reads top-down instead
```

## Stage 1 — Structure (structure.py)
1. Detect horizontal rule lines -> row boundaries `Y`
2. Calibrate column boundaries `X` from whichever reference row (header
   or footer) has the most *regular* spacing (guards against noise being
   mistaken for extra columns)
3. Build the full logical grid — every `(row, col)` gets a `CellRegion`.
   Row/column count is fixed here and never reduced by merges later.
4. For every adjacent cell pair, test whether a real wall (line) exists
   between them (`right_wall`, `bottom_wall`)
5. Union-Find over the grid: cells with no wall between them join the
   same block. This is a general 2D method — it catches vertical-only
   merges, horizontal-only merges, and combined rectangular blocks, all
   from the same logic, without assuming a shape in advance.
6. Each connected component becomes a `MergeRegion` (row range x column
   range). Purely geometric — no text has been read yet.

## Stage 2 — OCR + value decision (ocr_targeted.py)
For every `MergeRegion`:
1. OCR each row's own slice within the block *individually* first.
2. Decide whether the rows share one true value or hold genuinely
   distinct per-row data:
   - 0–1 rows have real (non-noise) text -> one true value, centered in
     an otherwise-blank block -> propagate it.
   - A trailing number appears on a **majority** of the substantial
     rows -> one true shared value; minority noisy misreads are
     overruled, not treated as distinct data.
   - Every row has its own unique trailing number, no repeats -> these
     are genuinely distinct values (e.g. real sequential reference
     numbers) -> each row keeps its own OCR result.
   - Otherwise, fall back to strict (0.90) whole-string similarity.
3. Standalone (non-merged) cells are OCR'd independently.

This majority-vote step exists specifically to avoid two opposite
failure modes seen during development: (a) blindly propagating one
OCR read across a whole block even when rows actually hold distinct
data, and (b) fracturing one true shared value into fake "distinct"
values just because OCR noise misread a digit on one row.

## Stage 2.5 — Header discovery (header.py, header_vocab.py)
No new OCR pass -- Stage 2 already read every cell in the grid,
including the header row. This stage just interprets that row:
1. `struct.header_row_idx` says which grid row is the header (bottom
   row by default). Read that row's already-OCR'd text, one cell per
   column.
2. Fuzzy-match each cell's text against `header_vocab.COLUMN_ALIASES`
   -- a bank of known column concepts and their common phrasings/OCR
   misreads (e.g. `"OTY"` / `"Quantity"` -> `"QTY"`). A close match
   snaps to the canonical name so `postprocess.py`'s corrector for that
   concept still applies.
3. No match close enough -- keep the cleaned raw OCR text as the column
   name instead of forcing it into the wrong bucket or dropping it.
   This is what keeps the pipeline generalized to sheets with columns
   this vocabulary hasn't seen before, rather than requiring every
   table to match one fixed 8-column schema.
4. Column *count* was already decided by Stage 1's geometry -- this
   stage only names whatever columns Stage 1 found.
5. Because the header sits at the bottom, the row directly above it is
   logically row 1 -- `build_header()` also returns `data_row_order`,
   the reverse of the image's own row order, so Stage 3 writes rows out
   in correct reading order instead of upside down.

## Stage 3 — Excel export (export_xlsx.py)
- **Vertical merges** (same column, multiple rows): never use Excel's
  real cell merge. Every row stays its own independent row; the shared
  value is repeated into each row's own cell.
- **Horizontal merges** (same row, multiple columns): real
  `merge_cells()`, one visual span per affected row.
- **Combined blocks** (multi-row + multi-column): split per row — each
  row gets its own horizontal merge across the block's columns; rows
  are never merged with each other.

## Known limitations
- Line/wall detection depends on scan quality; thresholds
  (`line_presence_thresh`, `row_frac_thresh`, `col_frac_thresh` in
  `TableStructure.__init__`) may need retuning per document batch.
- OCR accuracy is capped by source image resolution — this was tested
  against low-resolution samples where recognition, not structure, was
  the binding constraint.
- The majority-vote decision is a heuristic, not a guarantee — very
  small merge blocks (2 rows) with no clear majority, or noise that
  corrupts most rows rather than a minority, can still go either way.
