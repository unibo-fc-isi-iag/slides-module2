"""Generates the quantization figures of content/llmaas: python sources/llmaas/quantization-figures.py content/llmaas"""
import math, random, sys

OUT = sys.argv[1]
FONT = "'Signika Negative', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Menlo, Consolas, monospace"
INK, GREY, LGREY = "#222222", "#555555", "#8a8f98"
BLUE, LBLUE = "#2690e7", "#d6e8fb"
GREEN, LGREEN = "#3a9e4f", "#dcf0e0"
ORANGE, LORANGE = "#e8871e", "#fbe7cf"
RED = "#d64545"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []

    def add(self, s):
        self.el.append(s)

    def text(self, x, y, s, size=20, fill=GREY, anchor="middle", bold=False, mono=False, italic=False):
        f = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
                 f'{" font-weight=\"bold\"" if bold else ""}{f}{it}>{s}</text>')

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=2, rx=0, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=2, arrow=None, dash=None):
        m = f' marker-end="url(#{arrow})"' if arrow else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{m}{d}/>')

    def circle(self, x, y, r, fill, stroke=None):
        st = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{st}/>')

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" font-family="{FONT}">\n'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#222222"/></marker>'
                '<marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#8a8f98"/></marker></defs>\n')
        open(f"{OUT}/{name}", "w").write(head + "\n".join(self.el) + "\n</svg>\n")


def diverging(v, vmax):
    """blue (negative) - white - orange (positive)"""
    t = max(-1, min(1, v / vmax))
    rgb = (232, 135, 30) if t > 0 else (38, 144, 231)
    return "#%02x%02x%02x" % tuple(round(255 + (c - 255) * abs(t)) for c in rgb)


def quantize(ws, levels, lo=None, hi=None):
    lo = min(ws) if lo is None else lo
    hi = max(ws) if hi is None else hi
    s = (hi - lo) / (levels - 1)
    codes = [max(0, min(levels - 1, round((w - lo) / s))) for w in ws]
    return codes, [lo + c * s for c in codes], lo, s


# ---------------------------------------------------------------- 1. blocks of weights
def blocks():
    rnd = random.Random(7)
    R, C, cell = 20, 64, 9
    W = [[rnd.gauss(0, 0.02) for _ in range(C)] for _ in range(R)]
    g = SVG(1600, 560)
    x0, y0 = 40, 150
    g.text(x0 + C * cell / 2, 60, "a weight tensor (one matrix of a layer)", 28, INK, bold=True)
    g.text(x0 + C * cell / 2, 92, "e.g. 4096 × 4096 weights, here only a corner", 22)
    for r in range(R):
        for c in range(C):
            g.rect(x0 + c * cell, y0 + r * cell, cell, cell, diverging(W[r][c], 0.04))
    g.rect(x0, y0, C * cell, R * cell, stroke=LGREY, sw=1)
    hr = 8
    g.rect(x0, y0 + hr * cell, C * cell, cell, stroke=INK, sw=2)                 # the row
    g.rect(x0, y0 + hr * cell - 3, 32 * cell, cell + 6, stroke=GREEN, sw=4)      # one block
    g.rect(x0 + 32 * cell, y0 + hr * cell - 3, 32 * cell, cell + 6, stroke=GREEN, sw=2, extra=' stroke-dasharray="6 4"')
    g.text(x0, y0 + R * cell + 34, "one row = consecutive blocks of 32 weights", 20, anchor="start")
    g.text(x0, y0 + R * cell + 62, "colour = value (blue negative, orange positive)", 18, anchor="start")
    # zoom
    ws = W[hr][:32]
    codes, _, lo, s = quantize(ws, 16)
    hi = max(ws)
    bx, bw = 720, 26
    mid, k = 270, 2600  # zero line, px per unit
    g.text(bx + 16 * bw, 60, "one block: 32 weights", 28, INK, bold=True)
    g.text(bx + 16 * bw, 92, "bar height = weight value; quantized on its own", 22)
    g.line(x0 + 32 * cell, y0 + hr * cell - 3, bx - 10, mid - 110, LGREY, 1.5, dash="5 4")
    g.line(x0 + 32 * cell, y0 + hr * cell + cell + 3, bx - 10, mid + 110, LGREY, 1.5, dash="5 4")
    g.line(bx - 10, mid, bx + 32 * bw + 10, mid, INK, 2)
    for i, w in enumerate(ws):
        h = w * k
        g.rect(bx + i * bw + 3, mid - max(h, 0), bw - 6, abs(h), diverging(w, 0.04), stroke=LGREY, sw=1)
    g.text(bx + 32 * bw + 18, mid + 6, "0", 18, anchor="start")
    cy = 400
    g.text(bx - 18, cy + 25, "code", 18, anchor="end")
    for i, c in enumerate(codes):
        g.rect(bx + i * bw + 1, cy, bw - 2, 36, LGREEN, GREEN, 1.5, rx=4)
        g.text(bx + i * bw + bw / 2, cy + 24, str(c), 14, INK, mono=True)
    g.rect(bx + 4 * bw, 462, 24 * bw, 70, LORANGE, ORANGE, 2, rx=12)
    g.text(bx + 16 * bw, 490, f"stored once per block: min = {lo:.4f}, scale s = {s:.4f}", 21, INK)
    g.text(bx + 16 * bw, 518, "+ 32 × 4-bit codes (one per weight)", 21, INK)
    g.save("weight-blocks.svg")


# ---------------------------------------------------------------- 2. k-quants
def kquants():
    g = SVG(1600, 640)
    g.text(800, 50, "Q4_K: one super-block = 256 weights = 8 sub-blocks of 32", 28, INK, bold=True)
    x0, sw_ = 300, 150
    y = 120
    g.rect(40, y, 220, 110, LORANGE, ORANGE, 2, rx=12)
    g.text(150, y + 34, "per super-block", 20, INK, bold=True)
    g.text(150, y + 64, "d, dmin", 20, INK, mono=True)
    g.text(150, y + 92, "2 × 16 bits (fp16)", 18)
    for i in range(8):
        x = x0 + i * sw_
        g.rect(x, y, sw_, 50, (LBLUE, "#b9d7f6")[i % 2])
        g.text(x + sw_ / 2, y + 32, "32 × 4-bit codes", 16, INK)
        g.line(x + sw_ / 2, y + 54, x + sw_ / 2, y + 86, LGREY, 2, "ahs")
        g.rect(x + 8, y + 90, sw_ - 16, 64, LGREEN, GREEN, 1.5, rx=6)
        g.text(x + sw_ / 2, y + 115, f"scale{i}, min{i}", 15, INK, mono=True)
        g.text(x + sw_ / 2, y + 140, "6 + 6 bits", 15)
    g.rect(x0, y, 8 * sw_, 50, stroke=BLUE, sw=2)
    g.text(x0 + 4 * sw_, y + 190, "sub-block i:   ŵ = d · scale_i · code  −  dmin · min_i", 24, INK, mono=True)
    g.text(x0 + 4 * sw_, y + 222, "the sub-block scales/mins are themselves quantized (6 bits), using d and dmin as their own scale", 19)
    # bits-per-weight bars
    by, bx, unit = 440, 520, 170
    g.text(800, by - 26, "bits per weight", 22, INK, bold=True)
    rows = [("fp16 min + scale per 32 weights", 4, 32 / 32, "5.0"),
            ("Q4_K (6-bit min + scale per 32, fp16 per 256)", 4, 12 / 32 + 32 / 256, "4.5")]
    for j, (lab, codes, over, tot) in enumerate(rows):
        yy = by + j * 70
        g.text(bx - 16, yy + 30, lab, 19, INK, anchor="end")
        g.rect(bx, yy, codes * unit, 44, LGREEN, GREEN, 1.5)
        g.text(bx + codes * unit / 2, yy + 29, "codes: 4", 18, INK)
        g.rect(bx + codes * unit, yy, over * unit, 44, LORANGE, ORANGE, 1.5)
        g.text(bx + (codes + over) * unit + 14, yy + 29, f"= {tot}", 20, INK, anchor="start", bold=True)
    g.text(bx + 4 * unit + 2, by + 150, "← overhead (scales, mins)", 18, anchor="start")
    g.save("k-quants.svg")


# ---------------------------------------------------------------- 3. quantization mixes
def mixes():
    n = 32
    more = lambda i: i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2
    tensors = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
    col = {"Q4_K": LGREEN, "Q5_K": "#9fd3aa", "Q6_K": ORANGE}

    def ty(mix, t, i):
        if mix == "S":
            if t == "attn_v" and i < 4: return "Q5_K"
            if t == "ffn_down" and i < n // 8: return "Q5_K"
        else:
            if t in ("attn_v", "ffn_down") and more(i): return "Q6_K"
        return "Q4_K"

    g = SVG(1600, 600)
    cw, ch, x0 = 32, 26, 200
    for p, (mix, title) in enumerate((("S", "Q4_K_S (small mix)"), ("M", "Q4_K_M (medium mix)"))):
        y0 = 70 + p * 270
        g.text(x0, y0 - 18, title, 26, INK, anchor="start", bold=True)
        for r, t in enumerate(tensors):
            g.text(x0 - 12, y0 + r * ch + 19, t, 17, INK, anchor="end", mono=True)
            for i in range(n):
                g.rect(x0 + i * cw, y0 + r * ch, cw - 2, ch - 2, col[ty(mix, t, i)])
        # output layer
        ox = x0 + n * cw + 30
        g.rect(ox, y0, 60, 7 * ch - 2, col["Q6_K"])
        g.text(ox + 30, y0 + 7 * ch + 22, "output", 17, INK, mono=True)
        for i in (0, 8, 16, 24, 31):
            g.text(x0 + i * cw + cw / 2 - 1, y0 + 7 * ch + 22, str(i), 16)
    g.text(x0 + n * cw / 2, 585, "layer (here: a 32-layer model)", 20)
    lx = 1350
    for j, (k, lab) in enumerate((("Q4_K", "Q4_K (4.5 bpw)"), ("Q5_K", "Q5_K (5.5 bpw)"), ("Q6_K", "Q6_K (6.6 bpw)"))):
        g.rect(lx, 90 + j * 44, 30, 30, col[k])
        g.text(lx + 42, 112 + j * 44, lab, 20, INK, anchor="start")
    g.save("quant-mixes.svg")


# ---------------------------------------------------------------- 4. importance matrix
def best_fit(ws, imp, levels):
    """grid search of (min, scale) minimising the importance-weighted squared error"""
    lo0, hi0 = min(ws), max(ws)
    best = None
    for a in range(0, 41):
        for b in range(0, 41):
            lo = lo0 + (hi0 - lo0) * a / 80
            hi = hi0 - (hi0 - lo0) * b / 80
            if hi <= lo: continue
            _, q, _, _ = quantize(ws, levels, lo, hi)
            err = sum(i * (w - v) ** 2 for w, v, i in zip(ws, q, imp))
            if best is None or err < best[0]:
                best = (err, lo, hi)
    return quantize(ws, levels, best[1], best[2])


def imatrix_fig():
    ws = [-0.9, -0.62, -0.45, -0.31, -0.18, -0.07, 0.04, 0.13, 0.27, 0.38, 0.55, 0.82]
    imp = [1, 1, 9, 1, 1, 8, 1, 1, 1, 9, 1, 1]
    levels = 4
    g = SVG(1600, 620)
    g.text(220, 50, "1. calibration run", 26, INK, bold=True)
    g.rect(20, 80, 170, 110, LBLUE, BLUE, 2, rx=8)
    for j, t in enumerate(("sample text", "(Wikipedia,", "code, chat…)")):
        g.text(105, 115 + j * 26, t, 18, INK)
    g.line(195, 135, 242, 135, INK, 3, "ah")
    g.rect(250, 90, 170, 90, "#eeeeee", LGREY, 2, rx=8)
    g.text(335, 130, "full-precision", 18, INK)
    g.text(335, 154, "model", 18, INK)
    g.text(220, 232, "for each weight w_j, record how large", 19, INK)
    g.text(220, 257, "its input x_j usually is:", 19, INK)
    g.text(220, 287, "imp_j = mean(x_j²)", 21, INK, mono=True)
    bx, base = 58, 500
    g.text(220, 340, "importance of the weights of one block", 19, INK, bold=True)
    for j, v in enumerate(imp):
        g.rect(bx + j * 28, base - v * 14, 20, v * 14, ORANGE if v > 5 else "#f3c48f")
    g.line(bx - 6, base, bx + 12 * 28 - 4, base, INK, 2)
    g.text(220, base + 28, "j = 1 … 12", 18)
    g.line(450, 50, 450, 590, "#dddddd", 2)

    def panel(px, title, sub, imp_used):
        codes, q, lo, s = best_fit(ws, imp_used, levels)
        g.text(px + 270, 50, title, 26, INK, bold=True)
        g.text(px + 270, 84, sub, 19, INK, mono=True)
        top, bot = 120, 500
        y = lambda v: bot - (v + 1) / 2 * (bot - top)
        for c in range(levels):
            lv = lo + c * s
            g.line(px, y(lv), px + 500, y(lv), GREEN, 2, dash="8 5")
            g.text(px + 506, y(lv) + 6, f"code {c}", 16, GREEN, anchor="start")
        e = 0
        for j, (w, v) in enumerate(zip(ws, q)):
            xx = px + 25 + j * 41
            important = imp[j] > 5
            if abs(w - v) > 1e-9:
                g.line(xx, y(w), xx, y(v), RED, 4 if important else 2)
            g.circle(xx, y(w), 10 if important else 6, ORANGE if important else BLUE, INK if important else None)
            if important: e += (w - v) ** 2
        g.text(px + 250, 545, f"squared error on the 3 important weights: {e:.3f}", 19, INK)
        return e

    panel(480, "2a. plain fit", "min Σ (w_j − ŵ_j)²", [1] * 12)
    panel(1040, "2b. imatrix-weighted fit", "min Σ imp_j · (w_j − ŵ_j)²", imp)
    g.text(1000, 595, "dots: weights (orange = important) · dashed: the 4 levels of a 2-bit code · red: rounding error", 18)
    g.save("imatrix.svg")


# ---------------------------------------------------------------- 5. non-bell-shaped blocks
def shapes():
    rnd = random.Random(3)
    normal = [rnd.gauss(0, 1) for _ in range(32)]
    outlier = [rnd.gauss(0, 1) for _ in range(31)] + [9.0]
    bimodal = [rnd.gauss(-2.2, 0.45) for _ in range(16)] + [rnd.gauss(2.2, 0.45) for _ in range(16)]
    cases = [("bell-shaped (the usual case)", normal), ("one outlier", outlier), ("two clusters (bimodal)", bimodal)]
    g = SVG(1600, 560)
    pw, r = 480, 7
    errs = []
    for p, (title, ws) in enumerate(cases):
        px = 40 + p * 520
        codes, q, lo, s = quantize(ws, 16)
        rms = math.sqrt(sum((w - v) ** 2 for w, v in zip(ws, q)) / len(ws))
        spread = math.sqrt(sum(w * w for w in ws) / len(ws))
        used = len(set(codes))
        g.text(px + pw / 2, 50, title, 26, INK, bold=True)
        cw = pw / 16
        base = 400
        for c in range(16):
            g.rect(px + c * cw + 1, 110, cw - 2, base - 110, ("#f4f6f9", "#eaeef3")[c % 2])
        stack = [0] * 16
        for w, c in sorted(zip(ws, codes)):
            stack[c] += 1
            g.circle(px + c * cw + cw / 2, base - 4 - (stack[c] - 0.5) * (2 * r + 2), r, BLUE)
        g.line(px, base, px + pw, base, INK, 2)
        for c in range(0, 16, 3):
            g.text(px + c * cw + cw / 2, base + 26, str(c), 16, mono=True)
        g.text(px + pw - cw / 2, base + 26, "15", 16, mono=True)
        g.text(px + pw / 2, base + 52, "code (16 levels between min and max)", 18)
        g.text(px + pw / 2, 488, f"codes actually used: {used} / 16", 21, RED if used < 12 else INK, bold=used < 12)
        errs.append(rms)
        worse = rms > 1.5 * errs[0]
        g.text(px + pw / 2, 518, f"mean rounding error: {rms:.2f}" + (f"  (×{rms / errs[0]:.1f})" if p else ""), 21,
               RED if worse else INK, bold=worse)
    g.text(800, 90, "each dot = one of the 32 weights of a block (typical size ≈ 1), stacked on the code it gets", 19)
    g.save("non-bell-blocks.svg")


blocks(); kquants(); mixes(); imatrix_fig(); shapes()
