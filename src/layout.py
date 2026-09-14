"""
src/layout.py — LayoutLMv3 document region detection.

WHY THIS MODULE EXISTS:
Given a PDF page (words + bounding boxes + image), this module identifies
WHAT each region of the page IS — title, section header, paragraph, table,
list, caption. This structural understanding is the foundation of Novelty 1:
instead of chunking by character count, we chunk by semantic region.

Without this module, chunking is blind. With it, chunking respects meaning.

PIPELINE (per page):
    words ──group──▶ text lines ──LayoutLMv3──▶ labelled lines ──group──▶ regions

LEARN THIS — How LayoutLMv3 works (study this, it's interview gold):
LayoutLMv3 is a multimodal transformer that processes three input streams
simultaneously:
  1. TEXT stream    — the words on the page, tokenized
  2. LAYOUT stream  — the (x0, y0, x1, y1) box of each token,
                      normalized to a 0–1000 integer scale
  3. IMAGE stream   — the page as a grid of 16x16 patches (like ViT)
These three streams are fused inside the transformer attention layers,
so the model can reason: "this text says 'Table 1' AND it's in a bold
font AND it sits above a grid structure → this is a table caption."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch                         # torch: tensor operations for model I/O
from transformers import (           # transformers: HuggingFace model hub — the
    BatchEncoding,                   # standard library for pre-trained transformers;
    LayoutLMv3ForTokenClassification,# ships LayoutLMv3 + its processor, so we
    LayoutLMv3Processor,             # don't reimplement tokenization/patching.
)

from config import (
    LAYOUTLM_BBOX_SCALE,
    LAYOUTLM_CONFIDENCE_THRESHOLD,
    LAYOUTLM_DEFAULT_LABEL,
    LAYOUTLM_MAX_SEQ_LENGTH,
    LAYOUTLM_MODEL_NAME,
    LAYOUTLM_STRIDE,
    LINE_MAX_GAP_RATIO,
    LINE_Y_TOLERANCE,
    REGION_MAX_GAP_RATIO,
)
from src.ingestion import PageData, PageWord


# Region types whose text is spread out in 2D (table cells, labels inside a
# figure). Lines of these types may join a region side-by-side, not only
# from directly above — otherwise every table column becomes its own region.
_BLOCK_LABELS = {"table", "picture"}


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class TextLine:
    """
    One visual line of text — the unit LayoutLMv3 labels.

    WHY LINES, NOT WORDS:
    The checkpoint we use was fine-tuned on DocLayNet at LINE level: every
    token was given the bbox of its whole line. Feeding word-level boxes
    would be a train/test mismatch (like running a detector trained on
    224px crops on 1024px images). Labelling lines also means all words of
    a line always share one label — no "half a line is a table" noise.

    Fields:
        words:      The PageWords on this line, left to right.
        text:       Words joined by spaces.
        bbox:       [x0, y0, x1, y1] enclosing the line, in PDF points.
        page_num:   0-indexed page number.
        label:      Region type predicted by LayoutLMv3 (set after inference).
        confidence: Mean probability of that label over the line's tokens.
    """
    words: list[PageWord]
    text: str
    bbox: list[float]
    page_num: int
    label: str = LAYOUTLM_DEFAULT_LABEL
    confidence: float = 0.0

    @property
    def height(self) -> float:
        """Line height in PDF points (at least 1 to avoid divide-by-zero)."""
        return max(self.bbox[3] - self.bbox[1], 1.0)


@dataclass
class LayoutRegion:
    """
    A contiguous block of a page with one semantic label.

    WHY THIS EXISTS:
    A region is the atomic unit of chunking — Novelty 1 chunks AT region
    boundaries, so a paragraph, a table or a list is never cut in half.

    Fields:
        label:      Semantic type, e.g. "text", "title", "section_header",
                    "table", "list_item", "caption", "picture", "page_footer".
        words:      The PageWord objects that make up this region.
        text:       Region lines joined by newlines (top to bottom).
        bbox:       Bounding box of the entire region [x0, y0, x1, y1].
        page_num:   Which page this region is on.
        confidence: Mean confidence of the region's lines.
    """
    label: str
    words: list[PageWord]
    text: str
    bbox: list[float]      # [x0, y0, x1, y1] in original PDF point units
    page_num: int
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# LAYOUT DETECTOR (class-based to avoid reloading the model every call)
# ---------------------------------------------------------------------------

# WHY A CLASS (not just functions):
# Loading a transformer model takes 5–15 seconds. If detect() were a plain
# function that loaded the model each call, a 10-page PDF would spend
# 50–150 seconds just on model loading. Load once in __init__, reuse forever.

class LayoutDetector:
    """
    Wraps a LayoutLMv3 token-classification model for region detection.

    Usage:
        detector = LayoutDetector()
        regions = detector.detect_document(pages)   # all pages
        regions = detector.detect(pages[0])         # one page
    """

    def __init__(self, model_name: str = LAYOUTLM_MODEL_NAME) -> None:
        """
        Load the LayoutLMv3 processor and model from HuggingFace.

        LEARN THIS — What "processor" means in HuggingFace:
        A Processor bundles everything needed to turn raw inputs into model
        inputs: a tokenizer (text → token IDs), an image processor (page →
        resized, normalized pixel tensor) and bbox handling. We call
        processor(...) once and get a ready-to-run input dict.

        Args:
            model_name: HuggingFace hub id of a LayoutLMv3 token-classification
                checkpoint (must include processor/tokenizer files).
        """
        print(f"[LayoutDetector] Loading model: {model_name}")
        print("[LayoutDetector] First run downloads ~500MB — be patient.")

        # apply_ocr=False: we supply text + boxes ourselves (from pdfplumber).
        # apply_ocr=True would run Tesseract on the image — slower and less
        # accurate than the text already embedded in a digital PDF.
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        self.model.eval()   # disable dropout — inference, not training

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # id → clean label, e.g. 7 → "section_header"
        self.id2label: dict[int, str] = {
            int(i): _clean_label(name) for i, name in self.model.config.id2label.items()
        }
        print(f"[LayoutDetector] Ready on {self.device}. Labels: {list(self.id2label.values())}")

    def detect_document(self, pages: list[PageData]) -> list[LayoutRegion]:
        """
        Detect regions on every page of a document.

        Args:
            pages: PageData list from ingestion.load_pdf_pages().

        Returns:
            All regions, ordered by page, then top-to-bottom.
        """
        regions: list[LayoutRegion] = []
        for page in pages:
            regions.extend(self.detect(page))
        return regions

    def detect(self, page_data: PageData) -> list[LayoutRegion]:
        """
        Run LayoutLMv3 on one page and return detected regions.

        NAIVE APPROACH (don't do this):
            Split the raw text on blank lines and call each block a region.
            PROBLEM: a blank line can't tell a table from a paragraph from a
            heading — you get boundaries but no idea what each block IS.
        BETTER APPROACH (what we do):
            Label every line with LayoutLMv3 (text + position + pixels),
            then merge neighbouring lines that share a label into regions.

        Args:
            page_data: A PageData object from ingestion.py.

        Returns:
            Regions ordered top-to-bottom. A single fallback "text" region
            if the page has no extractable words (scanned page).
        """
        if not page_data.words:
            return _make_fallback_region(page_data)

        lines = group_words_into_lines(page_data.words)
        encoding = self._encode(lines, page_data)
        probs = self._predict(encoding)
        self._label_lines(lines, encoding, probs)
        return group_lines_into_regions(lines)

    def _encode(self, lines: list[TextLine], page_data: PageData) -> BatchEncoding:
        """
        Build model inputs, splitting long pages into overlapping windows.

        LEARN THIS — Bounding box normalization (0–1000 scale):
        LayoutLMv3 was trained with boxes as integers in [0, 1000], where
        1000 = full page width/height. pdfplumber gives PDF points (0–612 for
        US-letter width), so we rescale. Skip this and the model receives
        nonsense positions — region quality collapses.

        LEARN THIS — Overflow windows instead of truncation:
        The model sees at most 512 tokens; a dense page can have 800+.
        truncation alone would silently drop the bottom of the page.
        return_overflowing_tokens=True instead returns SEVERAL 512-token
        windows, each overlapping the previous one by `stride` tokens,
        and the processor repeats the page image once per window.

        Args:
            lines:     Text lines of the page (each treated as one "word").
            page_data: Source page (image + dimensions).

        Returns:
            BatchEncoding with one row per window. Keeps .word_ids(), which
            maps every token back to the index of the line it came from.
        """
        boxes = [_normalize_bbox(line.bbox, page_data.width, page_data.height) for line in lines]
        return self.processor(
            images=page_data.image.convert("RGB"),
            text=[line.text for line in lines],
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=LAYOUTLM_MAX_SEQ_LENGTH,
            stride=LAYOUTLM_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,   # the processor requires this with overflow
            return_tensors="pt",
        )

    def _predict(self, encoding: BatchEncoding) -> torch.Tensor:
        """
        Forward pass over all windows of a page.

        Args:
            encoding: Output of _encode().

        Returns:
            Probabilities of shape (num_windows, 512, num_labels), on CPU.
        """
        pixel_values = encoding["pixel_values"]
        if isinstance(pixel_values, list):   # overflow returns one image per window
            pixel_values = torch.stack([torch.as_tensor(p) for p in pixel_values])

        # Only pass what the model's forward() accepts — offset_mapping and
        # overflow_to_sample_mapping are bookkeeping, not model inputs.
        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "pixel_values": pixel_values,
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():   # no autograd graph → less memory, faster
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu()

    def _label_lines(
        self,
        lines: list[TextLine],
        encoding: BatchEncoding,
        probs: torch.Tensor,
    ) -> None:
        """
        Turn token-level probabilities into one label per line (in place).

        LEARN THIS — Token vs line labels:
        The tokenizer splits "hypertension" into several sub-word tokens, and
        a line into many more. The model predicts a label PER TOKEN. We must
        not assume "token i = line i" — after the first multi-token word that
        drifts out of alignment. Instead, encoding.word_ids() tells us which
        line each token came from (None for [CLS]/[SEP]/padding).
        We AVERAGE the probability vectors of all tokens of a line (across all
        windows it appears in), then take the argmax. Same idea as mean
        pooling in embedder.py: many noisy votes → one robust decision.

        Args:
            lines:    Lines to label (label/confidence are overwritten).
            encoding: Output of _encode() — provides the token→line mapping.
            probs:    Output of _predict().
        """
        line_ids = torch.tensor([
            [-1 if i is None else i for i in encoding.word_ids(w)]
            for w in range(probs.shape[0])
        ])
        mask = line_ids >= 0
        sums = torch.zeros(len(lines), probs.shape[-1])
        counts = torch.zeros(len(lines))
        sums.index_add_(0, line_ids[mask], probs[mask])
        counts.index_add_(0, line_ids[mask], torch.ones(int(mask.sum())))

        mean_probs = sums / counts.clamp(min=1).unsqueeze(1)
        scores, label_ids = mean_probs.max(dim=-1)
        for line, score, label_id, count in zip(lines, scores, label_ids, counts):
            confident = count > 0 and score >= LAYOUTLM_CONFIDENCE_THRESHOLD
            line.label = self.id2label[int(label_id)] if confident else LAYOUTLM_DEFAULT_LABEL
            line.confidence = float(score)


# ---------------------------------------------------------------------------
# WORDS → LINES
# ---------------------------------------------------------------------------

def group_words_into_lines(words: list[PageWord]) -> list[TextLine]:
    """
    Group pdfplumber words into visual text lines.

    pdfplumber.extract_words() already returns words row by row, left to
    right, so a new line starts whenever the next word is on a different
    row OR too far to the right (e.g. it belongs to the next column, or to
    the next table cell).

    Args:
        words: Words of one page, in pdfplumber order.

    Returns:
        TextLines in the same order. Empty list if words is empty.
    """
    if not words:
        return []

    lines: list[TextLine] = []
    current: list[PageWord] = [words[0]]
    for word in words[1:]:
        prev = current[-1]
        height = max(prev.y1 - prev.y0, 1.0)
        same_row = abs(word.y0 - prev.y0) <= LINE_Y_TOLERANCE
        gap = word.x0 - prev.x1
        if same_row and -height <= gap <= LINE_MAX_GAP_RATIO * height:
            current.append(word)
        else:
            lines.append(_make_line(current))
            current = [word]
    lines.append(_make_line(current))
    return lines


def _make_line(words: list[PageWord]) -> TextLine:
    """
    Build a TextLine from its words.

    Args:
        words: Non-empty list of words on one line.

    Returns:
        TextLine with text and enclosing bbox (label not yet assigned).
    """
    return TextLine(
        words=words,
        text=" ".join(w.text for w in words),
        bbox=_enclosing_bbox([[w.x0, w.y0, w.x1, w.y1] for w in words]),
        page_num=words[0].page_num,
    )


# ---------------------------------------------------------------------------
# LINES → REGIONS
# ---------------------------------------------------------------------------

def group_lines_into_regions(lines: list[TextLine]) -> list[LayoutRegion]:
    """
    Merge labelled lines into regions.

    NAIVE APPROACH (what the first version did):
        Walk lines in order and merge while the label stays the same.
        PROBLEM 1: every paragraph on a page is "text", so consecutive
        paragraphs — even in different columns — fuse into one giant region.
        PROBLEM 2: in a two-column page, rows alternate left/right column,
        so reading order ping-pongs between columns inside one region.
    BETTER APPROACH (what we do):
        Visit lines top to bottom. A line joins an existing region only if
        the label matches AND it sits directly below that region (overlaps
        it horizontally, small vertical gap). Otherwise it starts a new one.
        Columns stay separate; a large vertical gap starts a new block.

    Known limitation: this is a heuristic, not full reading-order recovery
    (e.g. the XY-cut algorithm). Unusual layouts can still merge oddly.

    Args:
        lines: Labelled lines of one page.

    Returns:
        LayoutRegions sorted top-to-bottom, then left-to-right.
    """
    groups: list[list[TextLine]] = []
    for line in sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0])):
        target = _find_region_to_extend(groups, line)
        if target is None:
            groups.append([line])
        else:
            target.append(line)

    regions = [_build_region(group) for group in groups]
    return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))


def _find_region_to_extend(
    groups: list[list[TextLine]],
    line: TextLine,
) -> list[TextLine] | None:
    """
    Find an open region that this line should be appended to.

    Args:
        groups: Regions built so far (each a list of lines).
        line:   The next line (top-to-bottom order).

    Returns:
        The matching group, or None if the line should start a new region.
    """
    for group in reversed(groups):            # most recent regions first
        if group[-1].label != line.label:
            continue
        x0, _, x1, y1 = _enclosing_bbox([ln.bbox for ln in group])
        vertical_gap = line.bbox[1] - y1      # negative = same row as region bottom
        if vertical_gap > REGION_MAX_GAP_RATIO * line.height:
            continue
        overlaps_x = line.bbox[0] < x1 and line.bbox[2] > x0
        if overlaps_x or line.label in _BLOCK_LABELS:
            return group
    return None


def _build_region(lines: list[TextLine]) -> LayoutRegion:
    """
    Construct a LayoutRegion from a group of same-label lines.

    Args:
        lines: Non-empty list of lines, top to bottom.

    Returns:
        LayoutRegion with joined text, enclosing bbox and mean confidence.
    """
    return LayoutRegion(
        label=lines[0].label,
        words=[w for ln in lines for w in ln.words],
        text="\n".join(ln.text for ln in lines),
        bbox=_enclosing_bbox([ln.bbox for ln in lines]),
        page_num=lines[0].page_num,
        confidence=sum(ln.confidence for ln in lines) / len(lines),
    )


# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------

def _normalize_bbox(bbox: list[float], page_width: float, page_height: float) -> list[int]:
    """
    Convert a PDF-point bbox to LayoutLMv3's 0–1000 integer scale.

    INTERVIEW ALERT — "Why normalize bounding boxes?"
    Answer: "Documents come in different page sizes. Normalizing makes box
    coordinates scale-invariant, so the model generalizes across A4, letter
    and custom sizes without retraining."

    Args:
        bbox:        [x0, y0, x1, y1] in PDF points.
        page_width:  Page width in PDF points.
        page_height: Page height in PDF points.

    Returns:
        [x0, y0, x1, y1] as integers clamped to [0, 1000].
    """
    scale = LAYOUTLM_BBOX_SCALE
    x0, y0, x1, y1 = bbox

    def clamp(val: float) -> int:
        return max(0, min(scale, int(val)))

    return [
        clamp(x0 / page_width * scale),
        clamp(y0 / page_height * scale),
        clamp(x1 / page_width * scale),
        clamp(y1 / page_height * scale),
    ]


def _enclosing_bbox(boxes: list[list[float]]) -> list[float]:
    """
    Smallest bbox containing all given boxes.

    Args:
        boxes: Non-empty list of [x0, y0, x1, y1].

    Returns:
        [min x0, min y0, max x1, max y1].
    """
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _clean_label(raw: str) -> str:
    """
    Normalize a model label: "Section-header" → "section_header".

    WHY NOT split on "-" (as the first version did):
    That trick strips BIO prefixes ("B-table" → "table"), but DocLayNet
    labels have no BIO prefix — splitting would turn "Section-header" into
    "header" and "List-item" into "item", corrupting the metadata.
    Lowercase + underscores gives ChromaDB-friendly, readable labels.

    Args:
        raw: Label string from model.config.id2label.

    Returns:
        Lowercase label with "-" and spaces replaced by "_".
    """
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _make_fallback_region(page_data: PageData) -> list[LayoutRegion]:
    """
    Create a single generic 'text' region for pages with no extractable words.

    WHY THIS EXISTS:
    Scanned pages have no words for LayoutLMv3 to label. Returning an empty
    list would silently drop the page; instead we keep one placeholder region
    (with raw_text, empty for true scans) so the page stays in the pipeline.
    The chunker skips it if its text is empty.

    Args:
        page_data: PageData with no words.

    Returns:
        A one-element list containing a fallback LayoutRegion.
    """
    return [LayoutRegion(
        label=LAYOUTLM_DEFAULT_LABEL,
        words=[],
        text=page_data.raw_text,
        bbox=[0.0, 0.0, page_data.width, page_data.height],
        page_num=page_data.page_num,
        confidence=0.0,
    )]
