"""
Build a PowerPoint / Google Slides template mirroring the astronomer talk.

One slide per slide of shape_talk_astronomers.tex: the title is set, the
figure is already placed where one exists, diagrams are drawn as editable
shapes, and every text area is a grey placeholder prompt. The full wording
from the LaTeX deck is in each slide's speaker notes, ready to paste.

Usage (from docs/slides/):
    python make_pptx_template.py            # writes shape_talk_template.pptx
Google Slides: File > Import slides, or upload the .pptx and open it.
"""

from __future__ import annotations

import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "shapes", "talk")
FIG_FULL = os.path.join(HERE, "..", "figures", "shapes")
OUT = os.path.join(HERE, "shape_talk_template.pptx")

# ---- palette (matches the figures) ----------------------------------------
INK = RGBColor(0x1F, 0x2A, 0x33)
BODY = RGBColor(0x3A, 0x42, 0x49)
MUTED = RGBColor(0x8A, 0x8A, 0x8A)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
ACCENT2 = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
PANEL = RGBColor(0xF3, 0xF6, 0xF9)
RULE = RGBColor(0xE3, 0xE8, 0xED)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TITLE_FONT = "Georgia"
BODY_FONT = "Arial"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
CONTENT_TOP = Inches(1.45)
CONTENT_BOTTOM = Inches(6.85)
FIG_H = Inches(2.9)


# ---- primitives -------------------------------------------------------------


def _text(
    slide,
    left,
    top,
    width,
    height,
    text,
    size=16,
    color=BODY,
    font=BODY_FONT,
    bold=False,
    italic=False,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    line_spacing=1.15,
):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = line
        f = r.font
        f.size = Pt(size)
        f.name = font
        f.bold = bold
        f.italic = italic
        f.color.rgb = color
    return tb


def _placeholder(slide, left, top, width, height, lines, size=16):
    """Grey italic prompt text; bullets get a dot prefix."""
    return _text(slide, left, top, width, height, lines, size=size, color=MUTED, italic=True)


def _bullets(slide, left, top, width, height, items, size=16, color=MUTED, italic=True):
    return _text(
        slide,
        left,
        top,
        width,
        height,
        [f"•  {t}" for t in items],
        size=size,
        color=color,
        italic=italic,
    )


def _rect(
    slide, left, top, width, height, fill=PANEL, line=None, rounded=True, dash=False, shape=None
):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, left, top, width, height)
    s.shadow.inherit = False
    if rounded and shape is None:
        s.adjustments[0] = 0.06
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE

            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return s


def _rule(slide, left, top, width, color=ACCENT, weight=3):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(weight))
    s.shadow.inherit = False
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    return s


def _picture_fit(slide, path, left, top, width, height):
    """Place an image inside a box, preserving aspect ratio, centred."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(width / iw, height / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(path, left + (width - w) // 2, top + (height - h) // 2, w, h)


def _figure_or_prompt(slide, name, left, top, width, height, prompt=None):
    path = os.path.join(FIG, name) if name else None
    if path and os.path.exists(path):
        return _picture_fit(slide, path, left, top, width, height)
    path = os.path.join(FIG_FULL, name) if name else None
    if path and os.path.exists(path):
        return _picture_fit(slide, path, left, top, width, height)
    box = _rect(slide, left, top, width, height, fill=PANEL, line=RULE, dash=True)
    _text(
        slide,
        left,
        top,
        width,
        height,
        prompt or f"[Insert figure: {name}]",
        size=14,
        color=MUTED,
        italic=True,
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    return box


# ---- slide chrome -----------------------------------------------------------


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title, notes="", kicker=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = WHITE
        if kicker:
            _text(
                s,
                MARGIN,
                Inches(0.32),
                Inches(8),
                Inches(0.3),
                kicker.upper(),
                size=10,
                color=ACCENT,
                bold=True,
            )
        _text(
            s,
            MARGIN,
            Inches(0.5),
            W - 2 * MARGIN,
            Inches(0.7),
            title,
            size=28,
            color=INK,
            font=TITLE_FONT,
        )
        _rule(s, MARGIN, Inches(1.22), Inches(1.6))
        # footer
        _text(
            s,
            MARGIN,
            Inches(7.02),
            Inches(6),
            Inches(0.3),
            "RTF  ·  Real-Time Fingerprints",
            size=9,
            color=MUTED,
        )
        _text(
            s,
            W - MARGIN - Inches(1),
            Inches(7.02),
            Inches(1),
            Inches(0.3),
            str(self.n),
            size=9,
            color=MUTED,
            align=PP_ALIGN.RIGHT,
        )
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self, path):
        self.prs.save(path)


# ---- layouts ----------------------------------------------------------------


def two_col(
    deck,
    title,
    fig,
    bullets,
    notes,
    fig_left=True,
    split=0.56,
    message=None,
    fig_prompt=None,
    kicker=None,
):
    s = deck.slide(title, notes, kicker)
    inner = W - 2 * MARGIN
    gap = Inches(0.4)
    fw = int(inner * split) - gap // 2
    tw = inner - fw - gap
    top = CONTENT_TOP
    if message:
        _text(s, MARGIN, top, inner, Inches(0.5), message, size=17, color=INK)
        top = top + Inches(0.6)
    h = CONTENT_BOTTOM - top
    fx = MARGIN if fig_left else MARGIN + tw + gap
    tx = MARGIN + fw + gap if fig_left else MARGIN
    _figure_or_prompt(s, fig, fx, top, fw, h, fig_prompt)
    _bullets(s, tx, top, tw, h, bullets)
    return s


def image_top(
    deck,
    title,
    fig,
    left_bullets,
    right_bullets,
    notes,
    left_head=None,
    right_head=None,
    fig_h=FIG_H,
    kicker=None,
):
    s = deck.slide(title, notes, kicker)
    inner = W - 2 * MARGIN
    _figure_or_prompt(s, fig, MARGIN, CONTENT_TOP, inner, fig_h)
    top = CONTENT_TOP + fig_h + Inches(0.25)
    h = CONTENT_BOTTOM - top
    colw = (inner - Inches(0.5)) // 2
    for i, (head, items) in enumerate(((left_head, left_bullets), (right_head, right_bullets))):
        x = MARGIN + i * (colw + Inches(0.5))
        y = top
        if head:
            _text(s, x, y, colw, Inches(0.35), head, size=15, color=INK, bold=True)
            y = y + Inches(0.4)
        _bullets(s, x, y, colw, h - (y - top), items, size=14)
    return s


# ---- the deck ---------------------------------------------------------------


def build():
    d = Deck()

    # 1. Title ---------------------------------------------------------------
    s = d.prs.slides.add_slide(d.blank)
    d.n += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    _rect(s, 0, 0, Inches(0.28), H, fill=ACCENT, rounded=False)
    _text(
        s,
        Inches(1.0),
        Inches(2.1),
        Inches(7.6),
        Inches(1.6),
        "Can a compressed alert still surprise us?",
        size=40,
        color=INK,
        font=TITLE_FONT,
    )
    _rule(s, Inches(1.0), Inches(3.75), Inches(1.6))
    _text(
        s,
        Inches(1.0),
        Inches(3.95),
        Inches(7.6),
        Inches(0.9),
        "Fingerprinting LSST alerts, and a test with circles and squares",
        size=20,
        color=BODY,
    )
    _placeholder(
        s,
        Inches(1.0),
        Inches(5.3),
        Inches(7),
        Inches(0.9),
        ["[Speaker names]", "[Institution  ·  Date]"],
        size=14,
    )
    _figure_or_prompt(
        s, "talk_stamp_triplet.png", Inches(8.9), Inches(2.4), Inches(3.9), Inches(2.7)
    )
    s.notes_slide.notes_text_frame.text = (
        "Say in one breath: this is a status report on an experiment inside a "
        "larger compression project, and the result is about what the "
        "compression forgets."
    )

    # 2. Problem -------------------------------------------------------------
    s = d.slide(
        "The problem: ten million alerts a night",
        notes=(
            "LSST will issue ~10 million alerts every night. Each one: the light "
            "curve so far, a 63x63 pixel cutout triplet, and catalogue metadata. "
            "Brokers filter the stream down to what they can serve and discard "
            "the rest. What you did not know to look for never reaches you."
        ),
        kicker="Background",
    )
    _bullets(
        s,
        MARGIN,
        CONTENT_TOP,
        Inches(6.0),
        Inches(4.5),
        [
            "[How many alerts per night]",
            "[What one alert contains]",
            "[What brokers do with the stream]",
            "[The thing that gets lost  — make this the punchline]",
        ],
    )
    # bandwidth bar diagram, editable
    bx, by = Inches(7.3), CONTENT_TOP + Inches(0.3)
    _rect(s, bx, by, Inches(5.4), Inches(0.75), fill=RULE, rounded=False)
    _text(
        s,
        bx + Inches(0.15),
        by,
        Inches(5),
        Inches(0.75),
        "[raw stream  ~100 MB/s]",
        size=14,
        color=BODY,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    _rect(s, bx, by + Inches(1.0), Inches(0.06), Inches(0.75), fill=ACCENT, rounded=False)
    _text(
        s,
        bx + Inches(0.2),
        by + Inches(1.0),
        Inches(5.2),
        Inches(0.75),
        "[what can be served  ~1 MB/s  (one HD video)]",
        size=14,
        color=BODY,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    _text(
        s,
        bx,
        by + Inches(2.0),
        Inches(5.4),
        Inches(0.4),
        "[Optional: a real cutout triplet below]",
        size=12,
        color=MUTED,
        italic=True,
    )
    _figure_or_prompt(s, "talk_stamp_triplet.png", bx, by + Inches(2.4), Inches(5.4), Inches(2.1))

    # 3-5. Inputs ------------------------------------------------------------
    image_top(
        d,
        "What is in an alert, 1/3: the cutout triplet",
        "talk_triplet_explained.png",
        [
            "[Science: tonight's exposure, what it contains]",
            "[Template: deep stack of earlier images]",
            "[Difference: the pipeline's subtraction]",
        ],
        [
            "[Not a subtraction of these crops — made on full frames]",
            "[Template gives host context]",
            "[Science + template expose bad subtractions]",
        ],
        notes=(
            "Science: tonight's exposure, 63x63 pixels (~1 arcmin) around the "
            "candidate: transient, host, neighbours, sky. Template: a deep stack "
            "of earlier images of the same field, the sky before. Difference: "
            "the survey pipeline's subtraction, what changed.\n\nWhy all three? "
            "The difference is NOT science minus template of these crops: the "
            "pipeline makes it on full frames with PSF matching and resampling "
            "the crops cannot reproduce. The template gives host context: "
            "nuclear, offset, on a star, empty sky. Science + template together "
            "expose bad subtractions."
        ),
        right_head="Why all three?",
        fig_h=Inches(2.6),
        kicker="Background",
    )
    two_col(
        d,
        "What is in an alert, 2/3: the light curve",
        "talk_light_curve.png",
        [
            "[What a light curve is, in one sentence]",
            "[Why the sampling is irregular]",
            "[What the model receives per detection — small table]",
            "[Note: switched off in today's toy]",
        ],
        notes=(
            "Brightness vs. time in three filters, sampled whenever the survey "
            "happened to look. Rise time, decline rate and colour evolution "
            "separate the classes.\n\nWhat the model receives, per detection: "
            "time since first detection log(1+dt); gap since previous one "
            "log(1+dt_prev); flux and flux error (log); filter g/r/i flag. A "
            "variable-length list of such rows; the two time features exist "
            "because the sampling is irregular. In today's toy this input is "
            "switched off."
        ),
        kicker="Background",
    )
    two_col(
        d,
        "What is in an alert, 3/3: the metadata",
        "talk_metadata_annotated.png",
        [
            "[~30 numbers the pipeline already computed]",
            "[Nearest catalogued source: distance, brightness, star/galaxy]",
            "[How point-like the detection is]",
            "[Real-vs-bogus score; detections so far]",
            "[Pixels say what it looks like; metadata says what catalogues know]",
        ],
        notes=(
            "About 30 numbers the pipeline has already computed for every alert, "
            "mostly about context: nearest catalogued source (distance, "
            "brightness, star or galaxy); how point-like the detection is; the "
            "survey's own real-vs-bogus score; how many detections so far. The "
            "pixels say what it looks like; the metadata says what the "
            "catalogues already know about that spot. Also switched off in "
            "today's toy."
        ),
        kicker="Background",
    )

    # 6. Idea ------------------------------------------------------------------
    s = d.slide(
        "The idea: a 256-byte fingerprint per alert",
        notes=(
            "Pipeline: alert (light curve + cutouts) -> encoder (a trained "
            "network) -> fingerprint (64 numbers = 256 bytes).\n\nThe analogy "
            "you already use: a full spectrum -> (colour, magnitude) on a CMD. "
            "Two numbers, chosen by physics, that preserve what matters. A "
            "fingerprint is the same idea with more numbers, chosen by training "
            "on 18,000 labelled ZTF transients.\n\nWhat you can do with "
            "fingerprints: classify (87% accuracy from 256 bytes; full pipeline "
            "94%); find look-alikes (nearest neighbours in fingerprint space); "
            "flag outliers (things far from everything seen before)."
        ),
        kicker="The idea",
    )
    # pipeline diagram: three cards and two arrows
    cw, ch = Inches(3.2), Inches(1.2)
    y = CONTENT_TOP + Inches(0.1)
    xs = [Inches(1.3), Inches(5.05), Inches(8.8)]
    labels = [
        ("[alert]", "light curve + cutouts", PANEL),
        ("[encoder]", "a trained network", RGBColor(0xE4, 0xEE, 0xFA)),
        ("[fingerprint]", "64 numbers = 256 bytes", PANEL),
    ]
    for x, (head, sub, fill) in zip(xs, labels):
        _rect(s, x, y, cw, ch, fill=fill, line=ACCENT)
        _text(
            s,
            x,
            y + Inches(0.15),
            cw,
            Inches(0.5),
            head,
            size=16,
            color=INK,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        _text(
            s,
            x,
            y + Inches(0.6),
            cw,
            Inches(0.5),
            sub,
            size=13,
            color=MUTED,
            italic=True,
            align=PP_ALIGN.CENTER,
        )
    for x in (xs[0] + cw + Inches(0.12), xs[1] + cw + Inches(0.12)):
        a = _rect(
            s,
            x,
            y + Inches(0.42),
            Inches(0.32),
            Inches(0.36),
            fill=MUTED,
            shape=MSO_SHAPE.RIGHT_ARROW,
        )
        a.line.fill.background()
    top = y + ch + Inches(0.45)
    colw = (W - 2 * MARGIN - Inches(0.5)) // 2
    for i, (head, items) in enumerate(
        (
            (
                "The analogy you already use",
                [
                    "[Spectrum → (colour, magnitude) on a CMD]",
                    "[Two numbers chosen by physics]",
                    "[Fingerprint: more numbers, chosen by training on 18k ZTF transients]",
                ],
            ),
            (
                "What you can do with fingerprints",
                [
                    "[Classify — accuracy from 256 bytes vs. full pipeline]",
                    "[Find look-alikes by distance]",
                    "[Flag outliers]",
                ],
            ),
        )
    ):
        x = MARGIN + i * (colw + Inches(0.5))
        _text(s, x, top, colw, Inches(0.4), head, size=15, color=INK, bold=True)
        _bullets(
            s, x, top + Inches(0.45), colw, CONTENT_BOTTOM - top - Inches(0.45), items, size=14
        )

    # 7. Tension -------------------------------------------------------------
    s = d.slide(
        "The tension: recognise vs. discover",
        notes=(
            "Two cartoon fingerprint spaces. Left: trained hard to recognise; "
            "tidy clusters for the known classes, a new thing lands INSIDE one. "
            "Right: keeps more than it needs; looser clusters, the new thing "
            "sits ALONE.\n\nThe network keeps only what its training rewards. If "
            "the training question is 'SN Ia or TDE?', nothing rewards keeping "
            "'this looks like neither'. This is the question the rest of the "
            "talk tests."
        ),
        kicker="The idea",
    )
    pw, ph = Inches(5.2), Inches(3.3)
    for i, (head, inside) in enumerate(
        (
            ("[trained hard to recognise]", True),
            ("[keeps more than it needs]", False),
        )
    ):
        px = MARGIN + Inches(0.6) + i * (pw + Inches(1.0))
        py = CONTENT_TOP + Inches(0.45)
        _text(
            s,
            px,
            CONTENT_TOP,
            pw,
            Inches(0.4),
            head,
            size=15,
            color=INK,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        _rect(s, px, py, pw, ph, fill=PANEL, line=RULE)
        dots_a = [(0.9, 0.6), (1.2, 0.9), (1.5, 0.55), (0.8, 1.05), (1.3, 1.3), (1.0, 1.6)]
        dots_b = [(3.7, 2.1), (4.1, 2.4), (3.8, 2.7), (4.3, 2.0), (3.5, 2.45), (4.0, 2.85)]
        if not inside:
            dots_a = [(x + 0.1 * (j % 3), y + 0.25 * (j % 2)) for j, (x, y) in enumerate(dots_a)]
            dots_b = [(x - 0.15 * (j % 2), y + 0.2 * (j % 3)) for j, (x, y) in enumerate(dots_b)]
        for (x, y), col in [(p, ACCENT) for p in dots_a] + [(p, ACCENT2) for p in dots_b]:
            _rect(
                s,
                px + Inches(x),
                py + Inches(y),
                Inches(0.16),
                Inches(0.16),
                fill=col,
                shape=MSO_SHAPE.OVAL,
            )
        sx, sy = (1.15, 0.95) if inside else (3.9, 0.7)
        _rect(
            s,
            px + Inches(sx),
            py + Inches(sy),
            Inches(0.3),
            Inches(0.3),
            fill=AQUA,
            shape=MSO_SHAPE.STAR_5_POINT,
        )
        _text(
            s,
            px + Inches(0.6),
            py + Inches(1.95),
            Inches(1.6),
            Inches(0.3),
            "[SN Ia]",
            size=11,
            color=MUTED,
            italic=True,
            align=PP_ALIGN.CENTER,
        )
        _text(
            s,
            px + Inches(3.3),
            py + Inches(2.95),
            Inches(1.6),
            Inches(0.3),
            "[TDE]",
            size=11,
            color=MUTED,
            italic=True,
            align=PP_ALIGN.CENTER,
        )
        _text(
            s,
            px + Inches(1.6) if inside else px + Inches(2.4),
            py + Inches(0.35) if inside else py + Inches(1.1),
            Inches(2.8),
            Inches(0.35),
            "[new thing lands inside]" if inside else "[new thing sits alone]",
            size=12,
            color=AQUA,
            italic=True,
        )
    _bullets(
        s,
        MARGIN,
        CONTENT_TOP + Inches(4.0),
        W - 2 * MARGIN,
        Inches(1.4),
        [
            "[The network keeps only what its training rewards]",
            "[If the question is 'SN Ia or TDE?', nothing rewards 'neither']",
            "[This is the question the rest of the talk tests]",
        ],
        size=15,
    )

    # 8. Toy -------------------------------------------------------------------
    image_top(
        d,
        "Why a toy: circles and squares",
        "talk_stamps_row.png",
        [
            "[Test the image part alone on a problem whose answer we know]",
            "[Light-curve input switched off: fingerprint depends only on the cutout]",
        ],
        [
            "[Equal-area shapes, random size, position, brightness, noise]",
            "[If this fails, nothing downstream is meaningful]",
        ],
        notes=(
            "Before trusting this on real transients, test the image part alone "
            "on a problem whose answer we know. Light-curve input switched off: "
            "the fingerprint depends only on the cutout. Synthetic stamps: "
            "circles and squares of equal area, random size, position and "
            "brightness, plus noise. Nothing but shape distinguishes them. If "
            "the fingerprint cannot separate these, nothing downstream is "
            "meaningful."
        ),
        fig_h=Inches(2.2),
        kicker="A toy we can understand",
    )

    # 9. Two encoders -----------------------------------------------------------
    s = d.slide(
        "Two ways to build the image encoder",
        notes=(
            "From scratch: 0.25 million parameters, starts knowing nothing, "
            "learns only from our stamps. Zoobot: 8.5 million parameters, "
            "pre-trained on Galaxy Zoo morphology votes, weights frozen: used "
            "exactly as published.\n\nZoobot is a public model (Walmsley et "
            "al.). We never retrain its weights; we only train a small read-out "
            "that maps its output into our 16 numbers. A first-year student vs. "
            "someone who has classified galaxies for a decade. Both get the "
            "same homework."
        ),
        kicker="A toy we can understand",
    )
    cw, ch = Inches(5.4), Inches(2.5)
    y = CONTENT_TOP + Inches(0.2)
    for i, (head, lines, fill) in enumerate(
        (
            (
                "[From scratch]",
                ["[parameter count]", "[starts knowing nothing]", "[learns only from our stamps]"],
                PANEL,
            ),
            (
                "[Zoobot]",
                [
                    "[parameter count]",
                    "[pre-trained on Galaxy Zoo morphology]",
                    "[weights frozen — used as published]",
                ],
                RGBColor(0xE4, 0xEE, 0xFA),
            ),
        )
    ):
        x = MARGIN + Inches(0.35) + i * (cw + Inches(0.65))
        _rect(s, x, y, cw, ch, fill=fill, line=ACCENT)
        _text(
            s,
            x,
            y + Inches(0.2),
            cw,
            Inches(0.5),
            head,
            size=20,
            color=INK,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        _text(
            s,
            x + Inches(0.4),
            y + Inches(0.8),
            cw - Inches(0.8),
            ch - Inches(1.0),
            lines,
            size=14,
            color=MUTED,
            italic=True,
            align=PP_ALIGN.CENTER,
        )
    _bullets(
        s,
        MARGIN,
        y + ch + Inches(0.4),
        W - 2 * MARGIN,
        Inches(1.5),
        [
            "[Zoobot is public (Walmsley et al.); we never retrain it, only a small read-out]",
            "[Analogy: first-year student vs. decade-long galaxy classifier, same homework]",
        ],
        size=15,
    )

    # 10. Result 1 ---------------------------------------------------------------
    s = d.slide(
        "Result 1: from scratch works, but is hungry and fragile",
        notes=(
            "256 examples: 62%. About 1000 examples: 99%. Add modest noise and "
            "512 examples is no longer enough; 2048 fixes it. Rare transient "
            "classes do not come with 1000 labelled examples."
        ),
        kicker="Results",
    )
    inner = W - 2 * MARGIN
    colw = (inner - Inches(0.4)) // 2
    fh = Inches(3.6)
    _figure_or_prompt(s, "talk_data_efficiency.png", MARGIN, CONTENT_TOP, colw, fh)
    _figure_or_prompt(s, "talk_noise.png", MARGIN + colw + Inches(0.4), CONTENT_TOP, colw, fh)
    _bullets(
        s,
        MARGIN,
        CONTENT_TOP + fh + Inches(0.2),
        inner,
        Inches(1.5),
        [
            "[Accuracy at 256 vs. ~1000 examples]",
            "[What noise does at 512 vs. 2048 examples]",
            "[Rare classes do not come with 1000 labelled examples  — punchline]",
        ],
        size=15,
    )

    # 11. Result 2 ----------------------------------------------------------------
    two_col(
        d,
        "Result 2: Zoobot already knows",
        "talk_backbone.png",
        [
            "['Before' = random read-out on frozen Zoobot features, no training]",
            "[Accuracy before any training]",
            "[The data-efficiency argument in one picture]",
        ],
        notes=(
            "'Before' means a random read-out on top of frozen Zoobot features. "
            "No training on circles or squares whatsoever. It already separates "
            "them at 95%. This is the data-efficiency argument in one picture."
        ),
        kicker="Results",
    )

    # 12. Discovery test ----------------------------------------------------------
    image_top(
        d,
        "The discovery test: shapes it has never seen",
        "talk_novel_row.png",
        [
            "[Train on circles and squares only, then show four new shapes]",
            "[Q1: is this fingerprint far from every known one?  (score meaning)]",
        ],
        [
            "[Q2: grouped blindly, do new shapes get their own groups?]",
            "[Ceiling check: with labels, could a straight line separate all six?]",
        ],
        notes=(
            "Train on circles and squares only. Then show four new shapes and "
            "ask, without telling it: (1) Is this fingerprint far from every "
            "circle and square we trained on? Score: how often a new shape "
            "looks more unusual than a known one; 50% useless, 100% perfect. "
            "(2) If we group the fingerprints blindly, do the new shapes get "
            "their own groups? Ceiling check: if we had labels, could a "
            "straight line separate all six?"
        ),
        fig_h=Inches(2.3),
        kicker="Results",
    )

    # 13. Result 3 ------------------------------------------------------------------
    image_top(
        d,
        "Result 3: from scratch cannot; Zoobot can",
        "talk_tsne_pair.png",
        [
            "[Detection and separability numbers]",
            "[Triangles, rings, ellipses filed as 'circle', confidently]",
        ],
        [
            "[Detection and separability numbers]",
            "[Ring and cross own islands; triangle and ellipse close]",
        ],
        notes=(
            "From scratch: new-shape detection 68%; six-shape separability 43%. "
            "Triangles, rings and ellipses are all filed as 'circle', "
            "confidently.\n\nZoobot, no training: detection 99.8%; separability "
            "99.5%. Ring and cross form their own islands; triangle and ellipse "
            "sit close.\n\nSay explicitly: the panel is a 2-D picture of a "
            "16-number fingerprint; the numbers are computed on all 16."
        ),
        left_head="From scratch",
        right_head="Zoobot, no training",
        fig_h=Inches(3.3),
        kicker="Results",
    )

    # 14. The catch -----------------------------------------------------------------
    two_col(
        d,
        "The catch: teaching it to classify squashes the fingerprint",
        "talk_pca_pair.png",
        [
            "[Fingerprint collapses onto a line once the read-out learns circle vs. square]",
            "[Separability before → after; ring lands on circle]",
            "[New-shape detection holds up — why]",
            "[Classification rewards one direction; reconstruction recovers most of it]",
        ],
        notes=(
            "Once the read-out learns circle vs. square, the fingerprint "
            "collapses onto a line. Six-shape separability: 99.5% -> 85%. Ring "
            "now sits on top of circle. New-shape detection holds up (96%): "
            "each shape keeps its own stretch of the line. Classification "
            "rewards one direction and lets the others fade. Adding "
            "'reconstruct the image' recovers most of it (97%)."
        ),
        split=0.58,
        kicker="Results",
    )

    # 15. Takeaways -----------------------------------------------------------------
    s = d.slide(
        "What this means, and what is next",
        notes=(
            "Three takeaways: (1) Use a Galaxy Zoo pre-trained encoder for "
            "cutouts; from scratch is not viable for rare classes. (2) Be "
            "deliberate about how hard the fingerprint is pushed to classify; "
            "keep a reconstruction-type objective in the mix. (3) Measure "
            "discoverability on real alerts next: train without TDEs, then ask "
            "whether TDEs look unusual.\n\nHonest caveats: synthetic shapes are "
            "easy; real stamps have hosts, PSF blur and faint sources. Not yet "
            "tested whether an ordinary ImageNet-pre-trained network does as "
            "well; if it does, the win is 'pre-training', not 'Galaxy Zoo'."
        ),
        kicker="Takeaways",
    )
    cw = (W - 2 * MARGIN - Inches(0.8)) // 3
    ch = Inches(2.6)
    for i, prompt in enumerate(
        (
            "[Use a pre-trained morphology encoder for cutouts]",
            "[Be deliberate about classification pressure]",
            "[Measure discoverability on real alerts next]",
        )
    ):
        x = MARGIN + i * (cw + Inches(0.4))
        _rect(s, x, CONTENT_TOP, cw, ch, fill=PANEL, line=None)
        _text(
            s,
            x + Inches(0.3),
            CONTENT_TOP + Inches(0.2),
            Inches(1),
            Inches(0.7),
            str(i + 1),
            size=36,
            color=ACCENT,
            font=TITLE_FONT,
        )
        _text(
            s,
            x + Inches(0.3),
            CONTENT_TOP + Inches(1.0),
            cw - Inches(0.6),
            ch - Inches(1.2),
            prompt,
            size=15,
            color=MUTED,
            italic=True,
        )
    cy = CONTENT_TOP + ch + Inches(0.35)
    _rect(s, MARGIN, cy, W - 2 * MARGIN, CONTENT_BOTTOM - cy, fill=None, line=ACCENT2, dash=True)
    _text(
        s,
        MARGIN + Inches(0.3),
        cy + Inches(0.12),
        Inches(4),
        Inches(0.35),
        "Honest caveats",
        size=14,
        color=ACCENT2,
        bold=True,
    )
    _bullets(
        s,
        MARGIN + Inches(0.3),
        cy + Inches(0.5),
        W - 2 * MARGIN - Inches(0.6),
        CONTENT_BOTTOM - cy - Inches(0.6),
        [
            "[Synthetic shapes are easy; real stamps have hosts, PSF blur, faint sources]",
            "[ImageNet control not yet run — pre-training vs. Galaxy Zoo]",
        ],
        size=14,
    )

    # Backups -------------------------------------------------------------------------
    s = d.slide(
        "Backup: the seven fingerprint variants",
        notes=(
            "Rows: untrained small network; small network classify / "
            "reconstruct / both; Zoobot no training; Zoobot classify; Zoobot "
            "both. Columns: separability (labels), new-shape detection, blind "
            "grouping, dims used. Numbers are in docs/shape_baseline.md "
            "section 8."
        ),
        kicker="Backup",
    )
    rows = [
        (
            "fingerprint built by",
            "separability\n(labels)",
            "new-shape\ndetection",
            "blind\ngrouping",
            "dims\nused",
        ),
        ("untrained small network", "", "", "", ""),
        ("small network, classify", "", "", "", ""),
        ("small network, reconstruct", "", "", "", ""),
        ("small network, both", "", "", "", ""),
        ("Zoobot, no training", "", "", "", ""),
        ("Zoobot, classify", "", "", "", ""),
        ("Zoobot, both", "", "", "", ""),
    ]
    tbl = s.shapes.add_table(
        len(rows), 5, MARGIN + Inches(0.8), CONTENT_TOP, W - 2 * MARGIN - Inches(1.6), Inches(4.2)
    ).table
    tbl.columns[0].width = Inches(4.2)
    for c in range(1, 5):
        tbl.columns[c].width = Inches(1.75)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r == 0 else WHITE
            cell.text = val if val else "[0.00]"
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size = Pt(13)
            f.name = BODY_FONT
            f.bold = r == 0
            f.color.rgb = INK if (r == 0 or c == 0) else MUTED
            f.italic = not (r == 0 or c == 0)
    _text(
        s,
        MARGIN + Inches(0.8),
        CONTENT_TOP + Inches(4.4),
        Inches(10),
        Inches(0.4),
        "[2048 training stamps, 256 held-out per shape, seeds]",
        size=12,
        color=MUTED,
        italic=True,
    )

    s = d.slide(
        "Backup: how the numbers are defined",
        notes=(
            "Separability (with labels): 6-way logistic regression on the "
            "16-number fingerprint, 5-fold cross-validated accuracy. New-shape "
            "detection: mean distance to the 5 nearest circle/square training "
            "fingerprints; AUROC of new vs. known. Blind grouping: k-means with "
            "k=6; adjusted mutual information against the true shape. Dims "
            "used: participation ratio of the fingerprint covariance (16 = all "
            "directions equal, 1 = a line). The 2-D maps: PCA shows the two "
            "largest directions (honest, can hide structure); t-SNE folds all "
            "16 into the plane (shows structure, exaggerates gaps)."
        ),
        kicker="Backup",
    )
    _bullets(
        s,
        MARGIN,
        CONTENT_TOP,
        W - 2 * MARGIN,
        Inches(5),
        [
            "[Separability (with labels): …]",
            "[New-shape detection: …]",
            "[Blind grouping: …]",
            "[Dims used: …]",
            "[The 2-D maps — PCA vs. t-SNE: …]",
        ],
        size=16,
    )

    two_col(
        d,
        "Backup: is noise a visibility problem or a data problem?",
        "04_noise.png",
        [
            "[Blob brightness range]",
            "[2048 examples: perfect at 0.25, useful at 0.5]",
            "[Only at noise ≈ brightness does more data stop helping]",
        ],
        notes=(
            "Blob brightness is 0.6-1.4. With 2048 examples the from-scratch "
            "network is perfect at noise 0.25 and still useful at 0.5; only at "
            "noise ≈ brightness does more data stop helping."
        ),
        split=0.62,
        kicker="Backup",
    )

    d.save(OUT)
    return OUT


if __name__ == "__main__":
    print("saved", build())
