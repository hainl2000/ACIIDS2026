"""Full flow of the corrected joint model, drawn in the visual language of the paper's Fig. 1-5.

Writes figures/joint_fixed_flow.{pdf,svg,png}. Coordinates are in points (1/72 in); the canvas is
W x H points and is meant to be placed at text width in the LNCS layout.

Usage: python figures/make_joint_fixed_flow.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.expanduser(
    "~/.claude/skills/synced/3f1f6f91-2fd9-46f4-8027-8ee32cc58b78_08e88fb1-088a-4252-898f-8b93dc59f160/canvas-design/canvas-fonts")
for name in ["Outfit-Regular.ttf", "Outfit-Bold.ttf"]:
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": ["Outfit", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
})

W, H = 590, 326

C = {
    "lgm": "#c7dd7c", "lfm": "#7bb522", "lm": "#bfe0a8", "ldm": "#6cb04e",
    "dgm": "#f39a8b", "dfm": "#e9654f", "dm": "#fbc8a7",
    "mgm": "#bdb4e0", "mfm": "#8272c8", "mm": "#d9d1e9", "mdm": "#9e8bd1",
    "lmm": "#c97b58",
    "z_l": "#cbdc84", "z_d": "#dc969f", "z_m": "#a9b7e1",
    "re": "#e0a1a3", "mu": "#c6da6f", "lv": "#dfe6c2",
    "navy": "#2b3990", "op": "#575a5e", "dash": "#8fbcdb", "text": "#1f2328", "muted": "#5d6673",
    "panel": "#eaf5fb", "panel_edge": "#5aa9d6", "lin": "#4ea3d6", "relu": "#b3dcf1", "agg": "#2f86c3",
    "white": "#ffffff", "frame_fc": "#f2f5f9", "frame_ec": "#b6c2d0",
}
VIEWS = {  # entity -> (GIP tile, functional/semantic tile, fused tile, z color, labels)
    "lnc": ("lgm", "lfm", "lm", "z_l", ("LGM1", "LFM1", "LM")),
    "dis": ("dgm", "dfm", "dm", "z_d", ("DGM1", "DFM1", "DM")),
    "mi": ("mgm", "mfm", "mm", "z_m", ("MGM1", "MFM1", "MM")),
}
BLOCKS = [[("LM", "lm"), ("LDM", "ldm"), ("LMM", "lmm")],
          [("DLM", "ldm"), ("DM", "dm"), ("DMM", "mdm")],
          [("MLM", "lmm"), ("MDM", "mdm"), ("MM", "mm")]]


def darker(color, f=0.2):
    r, g, b = to_rgb(color)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


for _k in ("lgm", "lfm", "dgm", "dfm", "mgm", "mfm"):
    C[_k + "2"] = darker(C[_k], 0.28)


def label_color(color):
    r, g, b = to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else "#2a2f36"


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 72, H / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    # ---------------------------------------------------------------- primitives
    def text(self, x, y, s, size=4.2, color=None, weight="normal", ha="center", va="center", **kw):
        self.ax.text(x, y, s, fontsize=size, color=color or C["text"], fontweight=weight, ha=ha, va=va, **kw)

    def tile(self, x, y, s, color, label=None, size=3.3, lattice=True):
        """Similarity square with a faint white lattice (lower-left corner at x, y)."""
        self.ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=1.1",
                                         fc=C[color], ec=darker(C[color]), lw=0.35))
        if lattice:
            for t in (1 / 3, 2 / 3):
                self.ax.plot([x + t * s] * 2, [y + 0.6, y + s - 0.6], color="white", lw=0.3, alpha=0.6)
                self.ax.plot([x + 0.6, x + s - 0.6], [y + t * s] * 2, color="white", lw=0.3, alpha=0.6)
        if label:
            self.text(x + s / 2, y + s / 2, label, size=size, weight="bold", color=label_color(C[color]))

    def stack(self, x, y, w, h, n=3, d=1.8):
        """Thin offset outlines behind an object: n further layers (drawn before the object)."""
        for i in range(n, 0, -1):
            self.ax.add_patch(FancyBboxPatch((x + i * d, y - i * d), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                             fc="white", ec=C["dash"], lw=0.45))

    def block(self, x, y, cell, gap=0.55, labels=True, size=3.1):
        """3x3 heterogeneous adjacency (lower-left corner x, y)."""
        for r, row in enumerate(BLOCKS):
            for c, (lab, col) in enumerate(row):
                self.tile(x + c * (cell + gap), y + (2 - r) * (cell + gap), cell, col,
                          lab if labels else None, size=size, lattice=cell > 6)
        return 3 * cell + 2 * gap

    def bar(self, x, y, w, h, color, n=4, vertical=False):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                         fc=C[color], ec=darker(C[color], 0.25), lw=0.35))
        for i in range(1, n):
            if vertical:
                self.ax.plot([x, x + w], [y + i * h / n] * 2, color=darker(C[color], 0.25), lw=0.3)
            else:
                self.ax.plot([x + i * w / n] * 2, [y, y + h], color=darker(C[color], 0.25), lw=0.3)

    def diamond(self, cx, cy, r, label):
        self.ax.add_patch(Polygon([(cx - r, cy), (cx, cy + r), (cx + r, cy), (cx, cy - r)], closed=True,
                                  fc=C["op"], ec="none", joinstyle="round"))
        self.text(cx, cy, label, size=3.2, color="white", weight="bold")

    def opbox(self, x, y, w, h, label, size=3.0, fc=None, color="white"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                         fc=fc or C["op"], ec="none"))
        self.text(x + w / 2, y + h / 2, label, size=size, color=color, weight="bold")

    def plate(self, x, y, w, h, slant, label, size=3.0):
        """Slanted encoder / decoder plate as in Fig. 2 and Fig. 4."""
        self.ax.add_patch(Polygon([(x, y), (x + w, y), (x + w + slant, y + h), (x + slant, y + h)], closed=True,
                                  fc=C["op"], ec=C["op"], lw=1.2, joinstyle="round"))
        self.text(x + (w + slant) / 2, y + h / 2, label, size=size, color="white", weight="bold", rotation=0)

    def arrow(self, x0, y0, x1, y1, color=None, lw=0.55, ls="-", head=2.4):
        # head length / width are in points (mutation_scale = 1)
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head},head_width={head * 0.42}",
                                          mutation_scale=1, color=color or C["navy"], lw=lw, ls=ls,
                                          shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt", zorder=2))

    def path_arrow(self, pts, color=None, lw=0.55, ls="-", head=2.4):
        """Polyline whose last segment ends in an arrowhead."""
        xs, ys = zip(*pts[:-1])
        self.ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=color or C["navy"], lw=lw, ls=ls,
                     solid_joinstyle="miter", solid_capstyle="butt", dash_capstyle="butt")
        self.arrow(*pts[-2], *pts[-1], color=color, lw=lw, ls=ls, head=head)

    def curve_arrow(self, p0, c0, c1, p1, color=None, lw=0.55, ls="-", head=2.4):
        verts = [p0, c0, c1, p1]
        self.ax.add_patch(FancyArrowPatch(path=Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                                          arrowstyle=f"-|>,head_length={head},head_width={head * 0.42}",
                                          mutation_scale=1, color=color or C["navy"], lw=lw, ls=ls, zorder=2))

    def plus(self, cx, cy, r=3.2, lw=1.5):
        self.ax.plot([cx - r, cx + r], [cy, cy], color=C["navy"], lw=lw, solid_capstyle="butt")
        self.ax.plot([cx, cx], [cy - r, cy + r], color=C["navy"], lw=lw, solid_capstyle="butt")

    def bracket(self, x0, x1, y0, y1, tick=1.6):
        for x, s in ((x0, 1), (x1, -1)):
            self.ax.plot([x + s * tick, x, x, x + s * tick], [y0, y0, y1, y1], color=C["navy"], lw=0.55)

    def container(self, x, y, w, h, title=None, fc="none", ec=None, ls=(0, (2.4, 1.6)), lw=0.5):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                         fc=fc, ec=ec or C["dash"], lw=lw, ls=ls))
        if title:
            self.text(x + 6, y + h - 5, title, size=4.4, color=C["muted"], weight="bold", ha="left")

    def gate(self, cx, cy, s=9):
        """Sigmoid gate (LSTM-diagram style): small rounded square with a sigma."""
        self.ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle="round,pad=0,rounding_size=1.4",
                                         fc=C["agg"], ec="none", zorder=3))
        self.text(cx, cy, r"$\sigma$", size=3.8, color="white", zorder=4)

    def node(self, cx, cy, op, r=2.8):
        """Point-wise operation node: op = 'mul' (circle with x) or 'add' (circle with +)."""
        self.ax.add_patch(plt.Circle((cx, cy), r, fc="white", ec=C["navy"], lw=0.55, zorder=3))
        d = r * 0.55
        if op == "mul":
            self.ax.plot([cx - d, cx + d], [cy - d, cy + d], color=C["navy"], lw=0.55, zorder=4)
            self.ax.plot([cx - d, cx + d], [cy + d, cy - d], color=C["navy"], lw=0.55, zorder=4)
        else:
            self.ax.plot([cx - r * 0.7, cx + r * 0.7], [cy, cy], color=C["navy"], lw=0.6, zorder=4)
            self.ax.plot([cx, cx], [cy - r * 0.7, cy + r * 0.7], color=C["navy"], lw=0.6, zorder=4)

    def line(self, xs, ys, lw=0.55):
        self.ax.plot(xs, ys, color=C["navy"], lw=lw, solid_joinstyle="miter", solid_capstyle="butt", zorder=2)

    def frame(self, x, y, w, h, label):
        """Soft module frame (behind its contents) with a small label above its top-left corner."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3",
                                         fc=C["frame_fc"], ec=C["frame_ec"], lw=0.5, zorder=0))
        self.text(x + 2, y + h + 4.2, label, size=4.2, color=C["op"], weight="bold", ha="left")

    def save(self, stem):
        for ext in ("pdf", "svg", "png"):
            self.fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(self.fig)


def draw():
    cv = Canvas()
    T = 13  # tile size

    # ================================================================ PHASE 1 (offline)
    # deep topology fusion (getMGSiNet, Eq. 8-9), drawn as in the paper's Fig. 1:
    #   M_1 = G_1 + F_1 (normalised);  G_{k+1} = M_k x G_k,  F_{k+1} = M_k x F_k (matrix products);  ...  M_k, k = 1..4
    # then A_k = [M_k blocks || association blocks], and a GCN-AE on every A_k gives R_e^(k).
    P1 = 192                                               # bottom of the Phase 1 band
    cv.container(6, P1, 386, H - 6 - P1, "PHASE 1  ·  DEEP TOPOLOGY FUSION + GCN-AE")
    rows1 = {"lnc": 287, "dis": 253, "mi": 219}           # row centres
    t = 11                                                 # tile size of the stage cascade
    for ent, yc in rows1.items():
        g, f, fused, _, (lg, lf, lfused) = VIEWS[ent]
        gy, fy = yc + 3, yc - 3 - t                        # GIP tile above, functional / semantic tile below
        # stage 1: G_1 + F_1 -> M_1
        cv.tile(14, gy, t, g, lg, size=2.9)
        cv.tile(14, fy, t, f, lf, size=2.9)
        cv.text(19.5, yc, "+", size=4.6, color=C["navy"], weight="bold")
        cv.arrow(27.5, yc, 35.5, yc)
        cv.tile(37, yc - t / 2, t, fused, f"{lfused}1", size=2.9)
        # M_1 multiplies G_1 and F_1 -> G_2, F_2
        cv.line([48, 51], [yc, yc])
        cv.line([51, 51], [yc - t / 2 - 3, yc + t / 2 + 3])
        for lane, src, col, dy in ((yc + t / 2 + 3, lg, g, 4.4), (yc - t / 2 - 3, lf, f, -4.4)):
            cv.line([51, 53.4], [lane, lane])
            cv.node(55.6, lane, "mul", r=2.2)
            cv.text(55.6, lane + dy, src, size=2.4, color=C["navy"])
            cv.arrow(57.8, lane, 62.6, lane, head=2.0)
        cv.tile(63, gy, t, g + "2", lg[:-1] + "2", size=2.9)
        cv.tile(63, fy, t, f + "2", lf[:-1] + "2", size=2.9)
        cv.text(68.5, yc, "+", size=4.6, color=C["navy"], weight="bold")
        for ddx in (-2.3, 0, 2.3):                         # further stages
            cv.ax.add_patch(plt.Circle((80 + ddx, yc), 0.5, fc=C["navy"], ec="none"))
        cv.arrow(84.5, yc, 92.5, yc)
        for i in range(3, 0, -1):                          # M_1 .. M_4 collected: k = 4..2 behind, k in front
            cv.tile(94.5 + i * 1.4, yc - t / 2 + i * 1.4, t, fused, lattice=False)
        cv.tile(94.5, yc - t / 2, t, fused, f"{lfused}$_k$", size=2.9)
    cv.text(101, 205.5, r"$k = 1 \ldots 4$", size=4.0)
    ym1 = rows1["dis"]                                     # main row of Phase 1
    cv.plus(121, ym1)
    cv.bracket(130, 152, 210, 296)
    for (lab, col), yc in zip([("LDM", "ldm"), ("LMM", "lmm"), ("MDM", "mdm")], rows1.values()):
        cv.tile(134.5, yc - T / 2, T, col, lab)
    cv.arrow(156, ym1, 167, ym1)
    cv.stack(171, ym1 - 13, 25.6, 25.6, n=3, d=1.8)
    cv.block(171, ym1 - 13, 8.2, gap=0.5, size=2.6)
    cv.text(187, ym1 - 24.5, r"$A_1 \ldots A_4$", size=4.0)
    cv.frame(210, ym1 - 18, 117, 36, "GCN-AE")
    cv.arrow(204, ym1, 213, ym1)
    cv.plate(215, ym1 - 12, 12, 24, 4, "Encode", size=3.0)
    cv.arrow(233, ym1, 241, ym1)
    cv.bar(243, ym1 - 3, 22, 6, "re")
    cv.arrow(267, ym1, 275, ym1)
    cv.plate(277, ym1 - 12, 12, 24, 4, "Decode", size=3.0)
    cv.arrow(295, ym1, 303, ym1)
    cv.block(305, ym1 - 8.5, 5.6, gap=0.4, labels=False)
    # stored embeddings R_e^(1..4)
    cv.path_arrow([(254, ym1 + 3.5), (254, ym1 + 29), (365, ym1 + 29), (365, ym1 + 19.5)], ls=(0, (2.2, 1.4)), lw=0.5)
    re_bars = [ym1 + 11, ym1 + 3, ym1 - 5, ym1 - 13]
    for yy in re_bars:
        cv.bar(352, yy, 26, 5.5, "re")
    re_out = (378, re_bars[1] + 2.75)                       # R_e leaves Phase 1 here

    # ================================================================ PHASE 2 (joint)
    # Phase 2 sits D2 below its natural place, leaving a channel between the two phases (y = 172 ... 192)
    # through which mu travels to the prediction stage; R_e enters the prediction stage from above.
    D2 = -28
    y2 = lambda y: y + D2
    cv.container(6, 6, 386, P1 - 20 - 6, "PHASE 2  ·  JOINT END-TO-END TRAINING")
    pairs = {"lnc": y2(168), "dis": y2(117), "mi": y2(66)}
    for ent, yc in pairs.items():
        g, f, fused, zc, (lg, lf, lfused) = VIEWS[ent]
        for (col, lab), yy in (((g, lg), yc + 8.5), ((f, lf), yc - 8.5)):
            cv.tile(14, yy - T / 2, T, col, lab)
            cv.arrow(29, yy, 36.5, yy)
            cv.diamond(42.5, yy, 5.6, "GCN")
            cv.arrow(48.6, yy, 55, yy)
            cv.bar(57, yy - 3, 22, 6, col)
        cv.path_arrow([(79, yc + 8.5), (83.5, yc + 8.5), (83.5, yc), (88.5, yc)])
        cv.ax.plot([79, 83.5, 83.5], [yc - 8.5, yc - 8.5, yc], color=C["navy"], lw=0.55)
        cv.opbox(89.5, yc - 4.5, 21, 9, "Attention", size=3.1)
        cv.arrow(111.5, yc, 118, yc)
        cv.bar(120, yc - 3, 22, 6, zc)
        cv.arrow(144, yc, 152, yc)
        cv.tile(154, yc - T / 2, T, fused, lfused)
    ym = y2(117)                                           # main row of Phase 2
    cv.plus(178, ym)
    cv.bracket(186, 208, y2(52), y2(182))
    for (lab, col), yc in zip([("LDM", "ldm"), ("LMM", "lmm"), ("MDM", "mdm")], pairs.values()):
        cv.tile(190.5, yc - T / 2, T, col, lab)
    cv.arrow(212, ym, 220, ym)
    cv.block(222, ym - 16.5, 10.6, gap=0.6, size=3.0)
    cv.frame(263, ym - 21, 125, 41, "VGAE")
    cv.arrow(258, ym, 266, ym)
    cv.plate(268, ym - 14, 12, 28, 4.5, "Encode", size=3.0)
    cv.path_arrow([(286, ym), (289, ym), (289, ym + 7.5), (292.5, ym + 7.5)])
    cv.path_arrow([(289, ym), (289, ym - 7.5), (292.5, ym - 7.5)])
    cv.bar(294, ym + 4.5, 18, 6, "mu")
    cv.bar(294, ym - 10.5, 18, 6, "lv")
    cv.ax.plot([313.5, 316.5, 316.5], [ym + 7.5, ym + 7.5, ym], color=C["navy"], lw=0.55)
    cv.ax.plot([313.5, 316.5, 316.5], [ym - 7.5, ym - 7.5, ym], color=C["navy"], lw=0.55)
    cv.arrow(316.5, ym, 336, ym)
    cv.plate(338, ym - 14, 12, 28, 4.5, "Decode", size=3.0)
    cv.arrow(356.5, ym, 362.5, ym)
    cv.block(364, ym - 10.5, 6.6, gap=0.4, labels=False)
    # reconstruction target below the main row
    cv.curve_arrow((238, ym - 18), (250, ym - 47), (362, ym - 45), (374, ym - 12.5), ls=(0, (2.2, 1.4)), lw=0.5)

    # ================================================================ PREDICTION (LDAGM)
    # left half: LDAGM column (as in Fig. 5); right half: the aggregate module drawn in detail
    px, pw = 398, 186
    cx = px + 45                                           # column centre line
    dx = px + pw - 46                                      # centre of the aggregate-module detail
    cv.container(px, 6, pw, H - 12, None, fc=C["panel"])
    cv.text(px + 6, H - 11, "LINK PREDICTION", size=4.4, color=C["agg"], weight="bold", ha="left")
    # node features: mu + R_e^(1..4) (vertical feature strips as in Fig. 5), group centred on cx;
    # the two phases converge here: R_e from Phase 1 (above), mu from Phase 2 (through the channel)
    strip, step, fy, fh = 6.5, 9, 172, 40                  # strip width, pitch, bottom, height
    group_w = strip + 7 + 3 * step + strip                 # mu, gap with "+", four R_e strips
    gx = cx - group_w / 2
    mu_x, re_x = gx, gx + strip + 7
    cv.bar(mu_x, fy, strip, fh, "mu", n=5, vertical=True)
    cv.plus(mu_x + strip + 3.5, fy + fh / 2, r=2.4, lw=1.1)
    for i in range(4):
        cv.bar(re_x + i * step, fy, strip, fh, "re", n=5, vertical=True)
    re_c = re_x + (3 * step + strip) / 2
    cv.path_arrow([re_out, (re_c, re_out[1]), (re_c, fy + fh + 0.5)], lw=0.55)
    bus = P1 - 10                                          # mu channel between the two phases
    cv.path_arrow([(303, ym + 10.6), (303, bus), (mu_x - 0.4, bus)], lw=0.55)
    cv.arrow(cx, fy - 1, cx, fy - 11)
    # LDAGM column (LDAGM.py): embedding Linear -> hidden MLP -> gated aggregation -> Linear -> sigmoid
    wcol, hb, pitch = 50, 12, 26
    x0 = cx - wcol / 2
    tops = [fy - 13 - i * pitch for i in range(5)]
    for top, (label, fc, color) in zip(tops, [("Linear (embed)", "lin", "white"),
                                             ("Linear · ReLU · Linear", "relu", C["agg"]),
                                             ("Aggregate (gates)", "agg", "white"),
                                             ("Linear", "lin", "white"),
                                             ("Sigmoid", "agg", "white")]):
        cv.opbox(x0, top - hb, wcol, hb, label, size=3.4, fc=C[fc], color=color)
        cv.arrow(cx, top - hb, cx, top - pitch + 0.5)
    cv.text(cx, tops[-1] - pitch - 7.5, "SCORE", size=4.4, weight="bold", color=C["agg"])

    # aggregate module in detail, beside the column, with zoom lines from the "Aggregate (gates)" box
    base = tops[2] - 40                                    # bottom of the detail frame
    aggregate_detail(cv, dx, base)
    agg_top, agg_bottom = tops[2], tops[2] - hb
    for (ya, yb) in ((agg_top, base + 72), (agg_bottom, base)):
        cv.ax.plot([x0 + wcol, dx - 44], [ya, yb], color=C["panel_edge"], lw=0.45, ls=(0, (2.0, 1.6)), zorder=1.4)
    cv.save("joint_fixed_flow")


def aggregate_detail(cv, cx, base):
    """LSTM-style drawing of AggregateLayer (LDAGM.py), frame from y=base to base+72, centred on cx:
    f = AG_{i-1} * sigma(.)  (forget),  n = H_i * sigma(.)  (input),
    u = sigma([n || f])      (update),  AG_i = u * n + (1 - u) * f."""
    y = lambda dy: base + dy
    xl, xr = cx - 30, cx + 30                              # state lane (AG) and hidden lane (H)
    cv.ax.add_patch(FancyBboxPatch((cx - 44, base), 88, 72, boxstyle="round,pad=0,rounding_size=3",
                                   fc="white", ec=C["panel_edge"], lw=0.5, ls=(0, (2.4, 1.6)), zorder=1.5))
    cv.text(cx, y(67.5), "Aggregate", size=3.6, color=C["agg"], weight="bold")
    cv.text(xl, y(85.5), r"$AG_{i-1}$", size=3.6, color=C["navy"])
    cv.text(xr, y(85.5), r"$H_i$", size=3.6, color=C["navy"])
    # forget gate on the state lane, input gate on the hidden lane
    for lane, gx in ((xl, xl + 14), (xr, xr - 14)):
        cv.arrow(lane, y(81), lane, y(48.9), head=2.0)
        cv.line([lane, gx, gx], [y(65), y(65), y(62.6)])
        cv.gate(gx, y(58))
        side = 1 if lane < gx else -1
        cv.line([gx, gx], [y(53.5), y(46)])
        cv.arrow(gx, y(46), lane + side * 2.9, y(46), head=2.0)
        cv.node(lane, y(46), "mul")
        cv.arrow(lane, y(43.2), lane, y(16.9), head=2.0)
    # update gate takes both gated streams, then u / (1 - u) weight them
    cv.line([xl, cx], [y(36), y(36)])
    cv.line([xr, cx], [y(36), y(36)])
    cv.arrow(cx, y(36), cx, y(31.6), head=2.0)
    cv.gate(cx, y(27))
    cv.line([cx, cx], [y(22.5), y(14)])
    cv.arrow(cx, y(14), xl + 2.9, y(14), head=2.0)
    cv.arrow(cx, y(14), xr - 2.9, y(14), head=2.0)
    cv.text(cx - 13, y(16.8), "1−", size=3.3, color=C["navy"])
    cv.node(xl, y(14), "mul")
    cv.node(xr, y(14), "mul")
    cv.line([xl, xl], [y(11.2), y(6)])
    cv.arrow(xl, y(6), cx - 3.2, y(6), head=2.0)
    cv.line([xr, xr], [y(11.2), y(6)])
    cv.arrow(xr, y(6), cx + 3.2, y(6), head=2.0)
    cv.node(cx, y(6), "add", r=3.2)
    cv.arrow(cx, y(2.8), cx, y(-10))
    cv.text(cx + 3, y(-5.5), r"$AG_i$", size=3.6, color=C["navy"], ha="left")


if __name__ == "__main__":
    draw()
    print("written:", ", ".join(f"figures/joint_fixed_flow.{e}" for e in ("pdf", "svg", "png")))
