"""Synthetic research documents shaped like a strategy report (the ODPD overnight study):
headings, prose, a table, a chart, and a code listing that crosses a page break."""

from __future__ import annotations

import base64

CODE = [
    "def causal_rank(x, min_obs=20):",
    "    out = []",
    "    for i, v in enumerate(x):",
    "        if i >= min_obs:",
    "            out.append(sum(h < v for h in x[:i]) / i)",
    "    return out",
    "",
    "def rule(event_next, wall_rank, pin_rank, gex):",
    "    return event_next or (wall_rank < 2 / 3 and pin_rank < 2 / 3 and gex > 0)",
]


def png(w: int = 200, h: int = 140) -> bytes:
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def make_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 60), "ODPD Overnight Dealer-Positioning Drift", fontsize=20, fontname="hebo")
    p1.insert_text((50, 90), "A long-or-flat SPY overnight strategy held from the close to the next open,",
                   fontsize=10, fontname="helv")
    p1.insert_text((50, 103), "only on nights with a tier-1 macro release or benign dealer positioning.",
                   fontsize=10, fontname="helv")
    p1.insert_text((50, 140), "1. Backtests", fontsize=15, fontname="hebo")
    # A ruled 3x3 table.
    x0, y0, cw, rh = 50, 160, 120, 18
    rows = [["id", "Sharpe", "Burke"], ["B0_always_long", "1.44", "1.59"], ["C6_event_or_wallpin_x_gex", "3.02", "4.76"]]
    for r in range(4):
        p1.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        p1.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            p1.insert_text((x0 + c * cw + 3, y0 + r * rh + 13), cell, fontsize=8, fontname="helv")
    p1.insert_image(pymupdf.Rect(50, 260, 250, 400), stream=png())
    p1.insert_text((50, 415), "Figure 1 - Cumulative overnight P&L of the final strategy.", fontsize=10, fontname="helv")
    p1.insert_text((50, 460), "2. Final backtest script", fontsize=15, fontname="hebo")
    p1.insert_text((50, 480), "The reproducible script is odpd_strategy.py (odpd_strategy.py).", fontsize=10, fontname="helv")
    y = 500
    for ln in CODE[:5]:
        indent = len(ln) - len(ln.lstrip())
        p1.insert_text((50 + indent * 4.8, y), ln.lstrip(), fontsize=8, fontname="cour")
        y += 11
    p1.insert_text((250, 820), "ODPD research report - page 1/3", fontsize=7, fontname="helv")
    p2 = doc.new_page()
    y = 60
    for ln in CODE[5:]:
        indent = len(ln) - len(ln.lstrip())
        if ln:
            p2.insert_text((50 + indent * 4.8, y), ln.lstrip(), fontsize=8, fontname="cour")
        y += 11
    p2.insert_text((50, 200), "3. Robustness", fontsize=15, fontname="hebo")
    p2.insert_text((50, 220), "Split halves give Sharpe 3.27 and 2.76; the placebo p-value is 0.008.", fontsize=10,
                   fontname="helv")
    p2.insert_text((250, 820), "ODPD research report - page 2/3", fontsize=7, fontname="helv")
    p3 = doc.new_page()
    p3.insert_text((50, 60), "Costs: Sharpe 1.54 at 5 bp per side.", fontsize=10, fontname="helv")
    p3.insert_text((250, 820), "ODPD research report - page 3/3", fontsize=7, fontname="helv")
    return doc.tobytes()


def make_html() -> bytes:
    img = base64.b64encode(png()).decode()
    code = "\n".join(CODE).replace("<", "&lt;")
    return f"""<!doctype html><html><head><title>ODPD research</title><style>body{{}}</style></head><body>
<h1>ODPD &mdash; Overnight Dealer-Positioning Drift</h1>
<p>A long-or-flat SPY <b>overnight</b> strategy that holds from the 15:59 print to the next 09:30 open.</p>
<h2>7. All backtests run</h2>
<table><caption>Overnight family</caption>
<tr><th>id</th><th>Sharpe</th><th>Burke</th></tr>
<tr><td>B0_always_long</td><td>1.444</td><td>1.592</td></tr>
<tr><td>C6_event_or_wallpin_x_gex</td><td>3.019</td><td>4.756</td></tr></table>
<figure><img src="data:image/png;base64,{img}" alt="equity curve"><figcaption>Figure 2 - Final strategy equity and drawdown.</figcaption></figure>
<h2>12. Final backtest script</h2>
<details open><summary>&#9660; odpd_strategy.py (odpd_strategy.py)</summary><pre><code>{code}</code></pre></details>
<script>var x = 1;</script>
</body></html>""".encode()
