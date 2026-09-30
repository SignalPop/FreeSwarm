"""Turning a research document (PDF, HTML, Markdown, text) into searchable pieces.

A strategy report is not one kind of content. Its argument is prose, its evidence is in
tables and charts, and its method is code -- and each is searched for differently: "what
did the overnight backtest return" wants the table, "how was wall position computed" wants
the code. So a document is parsed into typed **blocks**:

* ``text``   -- prose, grouped under the heading it sits in (the section path travels with it)
* ``table``  -- a table as GitHub markdown, with its caption
* ``figure`` -- an image (bytes kept for the console) with its caption; a vision model may
                add a description later (research.py), so a chart is found by what it shows
* ``code``   -- a code listing with its file name when the document gives one; indentation
                is rebuilt for PDFs, whose text layer keeps only each line's x position

``chunk()`` then merges consecutive prose of one section into retrieval-sized chunks and
leaves tables, figures and code whole. Nothing here touches the network, a model or the
database: bytes in, blocks out, so it is testable on its own.
"""

from __future__ import annotations

import base64
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser

CHUNK_CHARS = 1400          # prose chunk target: fits a small embedding model's 512-token window
MIN_FIGURE_PX = 120         # smaller embedded images are icons and rules, not figures
MAX_FIGURES = 200
CODE_FILE_RE = re.compile(r"\b([A-Za-z_][\w\-]*\.(?:py|sql|r|js|ts|cs|cpp|ipynb|sh|ps1|m|jl))\b", re.I)
CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|chart|table|exhibit)\s*[\d\w.\-–:]*", re.I)
PAGE_MARK_RE = re.compile(r"\bpage\s*\d+\s*(/|of)\s*\d+\b", re.I)
MONO_FONT_RE = re.compile(r"mono|courier|consol|menlo|monaco|inconsolata|code", re.I)


@dataclass
class Block:
    kind: str                     # text | table | figure | code
    text: str
    section: str = ""
    page: int | None = None
    label: str = ""               # code file name, figure/table caption
    image: bytes | None = None    # figure bytes (png/jpeg/gif/webp)
    image_ext: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Parsed:
    title: str
    kind: str                     # pdf | html | md | txt
    blocks: list[Block]
    pages: int | None = None
    outline: list[dict] = field(default_factory=list)   # [{level, title, page}]


# =======================================================================================
# Shared helpers
# =======================================================================================
def _norm_ws(s: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", s).strip()


def table_markdown(rows: list[list[str]], max_rows: int = 400) -> str:
    """Rows of cells -> a GitHub markdown table (first row is the header)."""
    rows = [[_norm_ws(str(c if c is not None else "")).replace("|", "\\|").replace("\n", " ") for c in r]
            for r in rows if r and any(str(c or "").strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows[:max_rows]]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def guess_language(code: str, label: str = "") -> str:
    ext = label.rsplit(".", 1)[-1].lower() if "." in label else ""
    if ext in ("py", "sql", "r", "js", "ts", "cs", "cpp", "sh", "ps1", "jl"):
        return {"py": "python"}.get(ext, ext)
    if re.search(r"^\s*(import |from \S+ import |def |class |@\w+|if __name__)", code, re.M):
        return "python"
    if re.search(r"^\s*(WITH|SELECT)\b", code, re.M | re.I) and re.search(r"\bFROM\b", code, re.I):
        return "sql"
    return ""


def _looks_like_code(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    hits = sum(bool(re.search(r"(^\s*(def|class|import|from|for|if|return|elif|else:|try:|except|with)\b)|[=(){}\[\]];?$|^\s*#", ln))
               for ln in lines)
    return hits / len(lines) > 0.5


def _file_label(*texts: str) -> str:
    for t in texts:
        m = CODE_FILE_RE.search(t or "")
        if m:
            return m.group(1)
    return ""


# =======================================================================================
# PDF
# =======================================================================================
def _is_mono(span: dict) -> bool:
    return bool(span.get("flags", 0) & 8) or bool(MONO_FONT_RE.search(span.get("font", "")))


def _furniture(pages_lines: list[list[str]]) -> set[str]:
    """Running headers/footers: a line (digits masked) on at least 40% of pages, or a page mark."""
    if len(pages_lines) < 3:
        return set()
    seen: Counter = Counter()
    for lines in pages_lines:
        seen.update({re.sub(r"\d+", "#", ln.strip()) for ln in lines if ln.strip()})
    return {k for k, n in seen.items() if n >= max(3, 0.4 * len(pages_lines)) and len(k) < 120}


def parse_pdf(data: bytes, filename: str = "") -> Parsed:
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    # Body text size = the size carrying the most characters; headings are clearly bigger.
    sizes: Counter = Counter()
    page_dicts = []
    for page in doc:
        d = page.get_text("dict", sort=True)
        page_dicts.append(d)
        for b in d["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    if sp["text"].strip() and not _is_mono(sp):
                        sizes[round(sp["size"], 1)] += len(sp["text"])
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    heading_sizes = sorted({s for s in sizes if s >= body * 1.25}, reverse=True)

    def level(size: float) -> int:
        for i, s in enumerate(heading_sizes[:3]):
            if size >= s - 0.05:
                return i + 1
        return 3

    furniture = _furniture([[ "".join(sp["text"] for sp in ln["spans"])
                              for b in d["blocks"] for ln in b.get("lines", [])] for d in page_dicts])

    items: list[tuple[int, float, str, dict]] = []   # (page, y, kind, payload)
    figures_left = MAX_FIGURES
    for pno, (page, d) in enumerate(zip(doc, page_dicts), start=1):
        exclude: list = []
        # Tables first: their text must not also come out as prose.
        try:
            for t in page.find_tables().tables:
                # Cell text by clip, not t.extract(): extract() drops underscores to a line of
                # their own ("B0 always long" + "_ _"), and identifiers are what reports tabulate.
                rows = [[page.get_textbox(pymupdf.Rect(c)).strip() if c else "" for c in r.cells] for r in t.rows]
                md = table_markdown(rows)
                if md and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    exclude.append(pymupdf.Rect(t.bbox))
                    items.append((pno, t.bbox[1], "table", {"md": md, "bbox": t.bbox, "rows": len(rows)}))
        except Exception:  # noqa: BLE001 -- a table finder failure must not lose the page's text
            pass
        # Embedded raster images: the charts of a report printed from HTML/matplotlib.
        for img in page.get_images(full=True):
            if figures_left <= 0:
                break
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
                pix = pymupdf.Pixmap(doc, xref)
                if pix.width < MIN_FIGURE_PX or pix.height < MIN_FIGURE_PX * 0.66:
                    continue
                if pix.n - pix.alpha >= 4:           # CMYK -> RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                png = pix.tobytes("png")
            except Exception:  # noqa: BLE001
                continue
            r = rects[0] if rects else pymupdf.Rect(0, 0, 0, 0)
            exclude.append(r)
            figures_left -= 1
            items.append((pno, r.y0, "figure", {"png": png, "bbox": tuple(r), "w": pix.width, "h": pix.height}))
        for b in d["blocks"]:
            for ln in b.get("lines", []):
                spans = [sp for sp in ln["spans"] if sp["text"]]
                if not spans:
                    continue
                text = "".join(sp["text"] for sp in spans)
                if not text.strip():
                    continue
                x0, y0, x1, y1 = ln["bbox"]
                if any(pymupdf.Rect(x0, y0, x1, y1).intersects(r) and
                       pymupdf.Rect(x0, y0, x1, y1).intersect(r).get_area() > 0.5 * pymupdf.Rect(x0, y0, x1, y1).get_area()
                       for r in exclude if not r.is_empty):
                    continue
                if re.sub(r"\d+", "#", text.strip()) in furniture or PAGE_MARK_RE.search(text):
                    continue
                mono = all(_is_mono(sp) for sp in spans if sp["text"].strip())
                size = max(sp["size"] for sp in spans)
                cw = max((sp["bbox"][2] - sp["bbox"][0]) / max(len(sp["text"]), 1) for sp in spans[:1])
                items.append((pno, y0, "line", {"text": text, "mono": mono, "size": size, "x0": x0, "y0": y0,
                                                "y1": y1, "cw": cw, "block": id(b),
                                                "bold": all(sp.get("flags", 0) & 16 for sp in spans)}))
    items.sort(key=lambda it: (it[0], it[1]))

    blocks: list[Block] = []
    outline: list[dict] = []
    path: list[tuple[int, str]] = []
    para: list[str] = []
    para_page = [None]
    code: list[dict] = []
    code_page = [None]
    last_text = [""]
    title = (doc.metadata or {}).get("title", "").strip()

    def section() -> str:
        return " > ".join(t for _, t in path)

    def flush_para() -> None:
        if para:
            txt = _norm_ws(re.sub(r"-\n(?=[a-z])", "", "\n".join(para)).replace("\n", " "))
            if txt:
                blocks.append(Block("text", txt, section(), para_page[0]))
                last_text[0] = txt
            para.clear()

    def flush_code() -> None:
        if code:
            base = min(c["x0"] for c in code)
            pitch = sorted(b["y0"] - a["y0"] for a, b in zip(code, code[1:]) if b["y0"] > a["y0"])
            pitch = pitch[len(pitch) // 2] if pitch else 0
            lines = []
            for k, c in enumerate(code):
                prev = code[k - 1] if k else None
                # An empty source line has no glyphs, only a bigger gap: put it back.
                if prev and pitch and prev["_page"] == c["_page"] and c["y0"] - prev["y0"] > 1.6 * pitch:
                    lines.append("")
                indent = int(round((c["x0"] - base) / c["cw"])) if c["cw"] > 0.5 else 0
                lines.append(" " * max(0, indent) + c["text"].rstrip())
            src = "\n".join(lines).strip("\n")
            label = _file_label(last_text[0][-300:], section().split(" > ")[-1] if path else "")
            blocks.append(Block("code", src, section(), code_page[0], label=label,
                                meta={"language": guess_language(src, label), "lines": len(lines)}))
            code.clear()

    prev_line: dict | None = None
    for pno, _y, kind, p in items:
        if kind == "line" and p["mono"]:
            flush_para()
            if not code:
                code_page[0] = pno
            p["_page"] = pno
            code.append(p)
            prev_line = p
            continue
        if kind == "line":
            text = p["text"].strip()
            is_heading = p["size"] >= body * 1.25 and len(text) < 160
            if is_heading:
                # A heading wraps onto several lines: continue the one just started.
                if prev_line is not None and prev_line.get("heading") and abs(prev_line["size"] - p["size"]) < 0.3 \
                        and prev_line["_page"] == pno and p["x0"] >= prev_line["x0"] - 2:
                    lvl, t = path[-1]
                    path[-1] = (lvl, f"{t} {text}")
                    outline[-1]["title"] = path[-1][1]
                    p["heading"], p["_page"] = True, pno
                    prev_line = p
                    continue
                flush_para()
                flush_code()
                lvl = level(p["size"])
                while path and path[-1][0] >= lvl:
                    path.pop()
                path.append((lvl, text))
                outline.append({"level": lvl, "title": text, "page": pno})
                p["heading"], p["_page"] = True, pno
                prev_line = p
                continue
            flush_code()
            # A new paragraph when the line starts a new PDF block or a visible gap opens.
            if para and prev_line is not None and not prev_line.get("heading") and (
                    p["block"] != prev_line.get("block") or pno != prev_line.get("_page")):
                flush_para()
            if not para:
                para_page[0] = pno
            para.append(text)
            p["_page"] = pno
            prev_line = p
            continue
        flush_para()
        # A code listing continues across a page break; anything else ends it.
        flush_code()
        if kind == "table":
            blocks.append(Block("table", p["md"], section(), pno, meta={"rows": p["rows"]}))
        elif kind == "figure":
            blocks.append(Block("figure", "", section(), pno, image=p["png"], image_ext="png",
                                meta={"bbox": p["bbox"], "width": p["w"], "height": p["h"]}))
        prev_line = None
    flush_para()
    flush_code()
    if not title and outline:
        title = outline[0]["title"]
    _caption_figures_and_tables(blocks)
    return Parsed(title=title or filename or "Untitled", kind="pdf", blocks=blocks, pages=len(doc), outline=outline)


def _caption_figures_and_tables(blocks: list[Block]) -> None:
    """Give each figure/table the caption next to it: a 'Figure ...' paragraph right after it
    (or right before, the usual table convention), else its section heading."""
    for i, b in enumerate(blocks):
        if b.kind not in ("figure", "table") or b.label:
            continue
        # A figure's caption sits below it; a table's above it, or directly below.
        order = (i + 1, i - 1, i + 2) if b.kind == "figure" else (i - 1, i + 1)
        near = [j for j in order if 0 <= j < len(blocks) and blocks[j].kind == "text"]
        cap = next((blocks[j].text for j in near if CAPTION_RE.match(blocks[j].text) and len(blocks[j].text) < 400), "")
        if not cap and b.kind == "figure":
            cap = next((blocks[j].text for j in near[:1] if len(blocks[j].text) < 200), "")
        b.label = cap[:400]
        if b.kind == "figure":
            b.text = "\n".join(x for x in (cap, f"Figure in section: {b.section}" if b.section else "") if x)


# =======================================================================================
# HTML
# =======================================================================================
_BLOCK_TAGS = {"p", "div", "li", "dd", "dt", "blockquote", "section", "article", "summary", "figcaption",
               "header", "footer", "main", "aside", "ul", "ol", "dl", "details", "figure", "br", "hr", "tr"}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "button", "select", "form", "iframe"}


class _Html(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.outline: list[dict] = []
        self.title = ""
        self._path: list[tuple[int, str]] = []
        self._skip = 0
        self._in_title = False
        self._heading: tuple[int, list[str]] | None = None
        self._text: list[str] = []
        self._pre: list[str] | None = None
        self._pre_class = ""
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_depth = 0
        self._caption: list[str] | None = None
        self._table_caption = ""
        self._summary: list[str] | None = None
        self._last_summary = ""
        self._figure_blocks: list[int] = []
        self._figcaption: list[str] | None = None
        self._last_text = ""

    # -- helpers -------------------------------------------------------------------------
    def section(self) -> str:
        return " > ".join(t for _, t in self._path)

    def flush(self) -> None:
        txt = _norm_ws("".join(self._text))
        self._text = []
        if txt:
            self.blocks.append(Block("text", txt, self.section()))
            self._last_text = txt

    # -- parser events -------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self.flush()
                self._table, self._table_caption = [], ""
            return
        if self._table is not None:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
            elif tag == "caption":
                self._caption = []
            elif tag == "br" and self._cell is not None:
                self._cell.append(" ")
            return
        if tag == "pre":
            self.flush()
            self._pre, self._pre_class = [], a.get("class") or ""
            return
        if self._pre is not None:
            if tag == "code" and a.get("class"):
                self._pre_class += " " + a["class"]
            if tag == "br":
                self._pre.append("\n")
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.flush()
            self._heading = (int(tag[1]), [])
            return
        if tag == "summary":
            self.flush()
            self._summary = []
            return
        if tag == "figure":
            self.flush()
            self._figure_blocks = []
        if tag == "figcaption":
            self.flush()
            self._figcaption = []
            return
        if tag == "img":
            self._image(a)
            return
        if tag in _BLOCK_TAGS:
            self.flush()
            if tag == "br":
                return

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._table is not None:
                md = table_markdown(self._table)
                if md:
                    cap = self._table_caption or (self._last_text if CAPTION_RE.match(self._last_text) else "")
                    self.blocks.append(Block("table", md, self.section(), label=cap[:400],
                                             meta={"rows": len(self._table)}))
                self._table = None
            return
        if self._table is not None:
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(_norm_ws("".join(self._cell)))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                if self._row:
                    self._table.append(self._row)
                self._row = None
            elif tag == "caption" and self._caption is not None:
                self._table_caption = _norm_ws("".join(self._caption))
                self._caption = None
            return
        if tag == "pre" and self._pre is not None:
            src = "".join(self._pre).strip("\n")
            self._pre = None
            if src.strip():
                label = _file_label(self._last_summary, self._last_text[-300:])
                m = re.search(r"language-(\w+)", self._pre_class)
                lang = m.group(1).lower() if m else guess_language(src, label)
                self.blocks.append(Block("code", src, self.section(), label=label,
                                         meta={"language": {"py": "python"}.get(lang, lang),
                                               "lines": src.count("\n") + 1}))
            return
        if self._pre is not None:
            return
        if self._heading and re.fullmatch(r"h[1-6]", tag):
            lvl, parts = self._heading
            text = _norm_ws("".join(parts))
            self._heading = None
            if text:
                while self._path and self._path[-1][0] >= lvl:
                    self._path.pop()
                self._path.append((lvl, text))
                self.outline.append({"level": lvl, "title": text, "page": None})
            return
        if tag == "summary" and self._summary is not None:
            self._last_summary = _norm_ws("".join(self._summary)).lstrip("▼▶►▾ ")
            self._summary = None
            if self._last_summary:
                self._last_text = self._last_summary
            return
        if tag == "figcaption" and self._figcaption is not None:
            cap = _norm_ws("".join(self._figcaption))
            self._figcaption = None
            for i in self._figure_blocks:
                self.blocks[i].label = cap[:400]
                self.blocks[i].text = "\n".join(x for x in (cap, self.blocks[i].text) if x)
            if cap and not self._figure_blocks:
                self.blocks.append(Block("text", cap, self.section()))
            return
        if tag == "figure":
            self.flush()
            self._figure_blocks = []
            return
        if tag in _BLOCK_TAGS:
            self.flush()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
        elif self._table is not None:
            if self._cell is not None:
                self._cell.append(data)
            elif self._caption is not None:
                self._caption.append(data)
        elif self._pre is not None:
            self._pre.append(data)
        elif self._heading is not None:
            self._heading[1].append(data)
        elif self._summary is not None:
            self._summary.append(data)
        elif self._figcaption is not None:
            self._figcaption.append(data)
        else:
            self._text.append(data)

    def _image(self, a: dict) -> None:
        src = a.get("src") or ""
        alt = _norm_ws(a.get("alt") or a.get("title") or "")
        data, ext = None, ""
        m = re.match(r"data:image/(png|jpe?g|gif|webp)[^;,]*;base64,(.*)", src, re.S | re.I)
        if m:
            try:
                data = base64.b64decode(re.sub(r"\s+", "", m.group(2)))
                ext = m.group(1).lower().replace("jpeg", "jpg")
            except ValueError:
                data = None
        if data is None and not alt:
            return   # an external/relative picture with nothing said about it: nothing to index
        if len(self.blocks) >= MAX_FIGURES * 4:
            return
        self.flush()
        self._figure_blocks.append(len(self.blocks))
        self.blocks.append(Block("figure", alt, self.section(), label=alt[:400], image=data, image_ext=ext,
                                 meta={"src": src if data is None else "", "alt": alt}))


def parse_html(data: bytes, filename: str = "") -> Parsed:
    text = data.decode("utf-8", errors="replace")
    m = re.search(r"<meta[^>]+charset=[\"']?([\w-]+)", text[:2000], re.I)
    if m and m.group(1).lower() not in ("utf-8", "utf8"):
        try:
            text = data.decode(m.group(1), errors="replace")
        except LookupError:
            pass
    p = _Html()
    p.feed(text)
    p.close()
    p.flush()
    blocks = [b for b in p.blocks if b.kind != "figure" or b.image or b.text]
    # Code a page shows without <pre> (a wide <div> of code) arrives as prose: recover it.
    for b in blocks:
        if b.kind == "text" and "\n" in b.text and _looks_like_code(b.text):
            b.kind, b.meta = "code", {"language": guess_language(b.text), "lines": b.text.count("\n") + 1}
    _caption_figures_and_tables(blocks)
    title = _norm_ws(p.title) or (p.outline[0]["title"] if p.outline else "") or filename or "Untitled"
    return Parsed(title=title, kind="html", blocks=blocks, outline=p.outline)


# =======================================================================================
# Markdown / plain text
# =======================================================================================
def parse_markdown(data: bytes, filename: str = "", kind: str = "md") -> Parsed:
    text = data.decode("utf-8", errors="replace")
    blocks: list[Block] = []
    outline: list[dict] = []
    path: list[tuple[int, str]] = []
    para: list[str] = []
    last = ""

    def sec() -> str:
        return " > ".join(t for _, t in path)

    def flush() -> None:
        nonlocal last
        if para:
            t = _norm_ws(" ".join(para))
            if t:
                blocks.append(Block("text", t, sec()))
                last = t
            para.clear()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        fence = re.match(r"^\s*(```|~~~)\s*([\w+-]*)", ln)
        if fence:
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence.group(1)):
                j += 1
            src = "\n".join(lines[i + 1:j])
            label = _file_label(last[-300:])
            blocks.append(Block("code", src, sec(), label=label,
                                meta={"language": fence.group(2).lower() or guess_language(src, label),
                                      "lines": j - i - 1}))
            i = j + 1
            continue
        h = re.match(r"^(#{1,6})\s+(.*)", ln) if kind == "md" else None
        if h:
            flush()
            lvl, t = len(h.group(1)), h.group(2).strip().strip("#").strip()
            while path and path[-1][0] >= lvl:
                path.pop()
            path.append((lvl, t))
            outline.append({"level": lvl, "title": t, "page": None})
            i += 1
            continue
        if kind == "md" and ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\s*\|?\s*:?-{2,}", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(Block("table", table_markdown(rows), sec(), label=last if CAPTION_RE.match(last) else "",
                                meta={"rows": len(rows)}))
            continue
        img = re.match(r"^\s*!\[([^\]]*)\]\(([^)]+)\)", ln) if kind == "md" else None
        if img:
            flush()
            blocks.append(Block("figure", img.group(1), sec(), label=img.group(1), meta={"src": img.group(2)}))
            i += 1
            continue
        if not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    title = (outline[0]["title"] if outline else "") or filename or "Untitled"
    return Parsed(title=title, kind=kind, blocks=[b for b in blocks if b.text or b.kind == "figure"], outline=outline)


# =======================================================================================
# Entry points
# =======================================================================================
KINDS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".xhtml": "html", ".md": "md", ".markdown": "md",
         ".txt": "txt"}


def kind_of(filename: str, data: bytes) -> str | None:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in KINDS:
        return KINDS[ext]
    head = data[:1024].lstrip().lower()
    if head.startswith(b"%pdf"):
        return "pdf"
    if head.startswith((b"<!doctype html", b"<html")):
        return "html"
    return None


def parse(data: bytes, filename: str) -> Parsed:
    kind = kind_of(filename, data)
    if kind == "pdf":
        return parse_pdf(data, filename)
    if kind == "html":
        return parse_html(data, filename)
    if kind in ("md", "txt"):
        return parse_markdown(data, filename, kind)
    raise ValueError(f"unsupported document type: {filename!r} (PDF, HTML, Markdown or text)")


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?;])\s+", text):
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = ""
        while len(sent) > limit:
            parts.append(sent[:limit])
            sent = sent[limit:]
        cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk(parsed: Parsed, limit: int = CHUNK_CHARS) -> list[Block]:
    """Prose merged per section into chunks of about `limit` characters; everything else whole."""
    out: list[Block] = []
    buf: list[Block] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        for piece in _split_long(text, limit):
            out.append(Block("text", piece, buf[0].section, buf[0].page))
        buf.clear()

    for b in parsed.blocks:
        if b.kind == "text":
            if buf and (b.section != buf[0].section or sum(len(x.text) for x in buf) + len(b.text) > limit):
                flush()
            buf.append(b)
        else:
            flush()
            out.append(b)
    flush()
    return out


def embed_text(b: Block, title: str) -> str:
    """What gets embedded for a chunk: the text itself, framed by where it sits. Code is
    embedded by what it DEFINES (names, docstrings, comments), which is what people search."""
    head = f"{title} | {b.section}" if b.section else title
    if b.kind == "code":
        names = re.findall(r"^\s*(?:def|class)\s+(\w+)", b.text, re.M)
        docs = re.findall(r'"""(.*?)"""', b.text, re.S)
        comments = re.findall(r"#\s*(.+)", b.text)
        body = " ".join([f"code {b.label}".strip(), "defines " + ", ".join(names[:40]) if names else "",
                         " ".join(d.strip() for d in docs[:6])[:800], " ".join(comments[:30])[:600]])
        return f"{head}\n{body}\n{b.text[:1200]}"
    if b.kind == "table":
        return f"{head}\ntable {b.label}\n{b.text[:1800]}"
    if b.kind == "figure":
        return f"{head}\nfigure {b.text}"
    return f"{head}\n{b.text}"
