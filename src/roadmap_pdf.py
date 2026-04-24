"""Beautifully designed placement roadmap PDF generator."""

from datetime import date
from fpdf import FPDF


# ─── Colour Palette ─────────────────────────────────────────────────────────
NAVY       = (15,  23,  42)
BLUE       = (37,  99, 235)
BLUE_LIGHT = (96, 165, 250)
PURPLE     = (109, 40, 217)
PURPLE_L   = (167, 139, 250)
TEAL       = (20, 184, 166)
TEAL_L     = (153, 246, 228)
GREEN      = (22, 163, 74)
GREEN_L    = (187, 247, 208)
AMBER      = (217, 119, 6)
AMBER_L    = (253, 230, 138)
RED        = (220, 38,  38)
RED_L      = (254, 202, 202)
WHITE      = (255, 255, 255)
GRAY_50    = (248, 250, 252)
GRAY_100   = (241, 245, 249)
GRAY_200   = (226, 232, 240)
GRAY_400   = (148, 163, 184)
GRAY_600   = (71,  85, 105)
GRAY_800   = (30,  41,  59)

PHASE_THEME = [
    {"header": BLUE,   "light": (239, 246, 255), "num": BLUE},
    {"header": PURPLE, "light": (245, 243, 255), "num": PURPLE},
    {"header": GREEN,  "light": (240, 253, 244), "num": GREEN},
]


def _s(text: str) -> str:
    """Sanitise text to Latin-1 for fpdf."""
    if not isinstance(text, str):
        text = str(text)
    MAP = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "*", "\u25cf": "*",
        "\u2026": "...", "\u2713": "OK", "\u2714": "OK",
        "\u00e9": "e",  "\u00e8": "e", "\u00e0": "a", "\u00fc": "u",
        "\u00e2": "a",  "\u00ea": "e", "\u00ee": "i", "\u00f4": "o",
    }
    for k, v in MAP.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class RoadmapPDF(FPDF):
    def header(self):
        pass  # drawn manually on page 1 only

    def footer(self):
        self.set_y(-14)
        self.set_fill_color(*NAVY)
        self.rect(0, self.h - 14, self.w, 14, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY_400)
        self.cell(
            0, 14,
            f"Placement Prediction System  |  Page {self.page_no()}  |  Confidential",
            align="C",
        )


# ─── Layout Helpers ──────────────────────────────────────────────────────────

def _lm(pdf):
    return pdf.l_margin

def _ew(pdf):
    return pdf.w - pdf.l_margin - pdf.r_margin


def _draw_header(pdf, student_name, field, today_str):
    """Full-width dark header with title, student info, and field badge."""
    header_h = 58

    # Background
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, header_h, "F")

    # Left accent stripe
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 5, header_h, "F")

    # Bottom accent bar
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, header_h, pdf.w, 3, "F")

    # Decorative right circle (subtle)
    pdf.set_fill_color(37, 99, 235)
    pdf.ellipse(pdf.w - 30, -20, 60, 60, "F")
    pdf.set_fill_color(109, 40, 217)
    pdf.ellipse(pdf.w - 15, 10, 35, 35, "F")
    # Re-draw right portion of header to cover overflow ellipses
    pdf.set_fill_color(*NAVY)
    pdf.rect(pdf.w - 0.5, 0, 1, header_h, "F")

    # Main title
    pdf.set_xy(10, 10)
    pdf.set_font("Helvetica", "B", 21)
    pdf.set_text_color(*WHITE)
    pdf.cell(_ew(pdf) - 5, 11, "Personalized Placement Roadmap", align="C")

    # Subtitle
    pdf.set_xy(10, 24)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(147, 197, 253)
    pdf.cell(
        _ew(pdf) - 5, 7,
        _s(f"Student: {student_name}   |   {today_str}"),
        align="C",
    )

    # Field badge
    badge_text = _s(f"  {field}  ")
    pdf.set_font("Helvetica", "B", 9)
    badge_w = pdf.get_string_width(badge_text) + 4
    badge_x = (pdf.w - badge_w) / 2
    badge_y = 34

    pdf.set_fill_color(*PURPLE)
    pdf.rect(badge_x, badge_y, badge_w, 11, "F")
    pdf.set_xy(badge_x, badge_y)
    pdf.set_text_color(*WHITE)
    pdf.cell(badge_w, 11, badge_text, align="C")

    # Reset cursor below header + stripe
    pdf.set_xy(_lm(pdf), header_h + 5)


def _section_title(pdf, text, icon_char, color):
    """Bold section title with colored left bar and icon."""
    pdf.ln(2)
    y = pdf.get_y()
    x = _lm(pdf)
    ew = _ew(pdf)

    # Left accent bar
    pdf.set_fill_color(*color)
    pdf.rect(x, y, 4, 9, "F")

    # Icon circle
    pdf.set_fill_color(*color)
    pdf.ellipse(x + 7, y + 0.5, 8, 8, "F")
    pdf.set_xy(x + 7, y + 0.5)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*WHITE)
    pdf.cell(8, 8, icon_char, align="C")

    # Title text
    pdf.set_xy(x + 18, y)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*color)
    pdf.cell(ew - 18, 9, _s(text))

    # Underline
    line_y = pdf.get_y()
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.4)
    pdf.line(x, line_y, x + ew, line_y)
    pdf.set_line_width(0.2)
    pdf.ln(5)


def _body(pdf, text, color=None):
    color = color or GRAY_800
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*color)
    pdf.set_x(_lm(pdf))
    pdf.multi_cell(_ew(pdf), 5.5, _s(text))
    pdf.ln(1)


def _bullet(pdf, text, color=None, symbol="->"):
    color = color or GRAY_800
    ew = _ew(pdf)
    indent = 8
    pdf.set_x(_lm(pdf) + indent)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*BLUE)
    pdf.cell(7, 5.5, symbol)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*color)
    pdf.multi_cell(ew - indent - 7, 5.5, _s(str(text)))
    pdf.set_x(_lm(pdf))


def _numbered(pdf, num, text, num_color=None):
    num_color = num_color or BLUE
    ew = _ew(pdf)
    indent = 8
    pdf.set_x(_lm(pdf) + indent)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*num_color)
    pdf.cell(8, 5.5, f"{num}.")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*GRAY_800)
    pdf.multi_cell(ew - indent - 8, 5.5, _s(str(text)))
    pdf.set_x(_lm(pdf))


def _phase_card(pdf, phase: dict, idx: int):
    """Draw one phase as a card with colored header + numbered actions."""
    theme = PHASE_THEME[idx % len(PHASE_THEME)]
    title  = _s(phase.get("title", f"Phase {idx+1}"))
    focus  = _s(phase.get("focus", ""))
    actions = phase.get("actions", [])

    x  = _lm(pdf)
    ew = _ew(pdf)
    y  = pdf.get_y()

    # Phase header strip
    pdf.set_fill_color(*theme["header"])
    pdf.rect(x, y, ew, 10, "F")

    # Phase number pill on the left
    pdf.set_fill_color(*WHITE)
    pdf.ellipse(x + 3, y + 1, 8, 8, "F")
    pdf.set_xy(x + 3, y + 1)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*theme["header"])
    pdf.cell(8, 8, str(idx + 1), align="C")

    # Phase title
    pdf.set_xy(x + 14, y)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*WHITE)
    pdf.cell(ew - 14, 10, title)
    pdf.ln(10)

    # Focus area (light tinted bg)
    if focus:
        pdf.set_fill_color(*theme["light"])
        pdf.rect(x, pdf.get_y(), ew, 8, "F")
        pdf.set_x(x + 4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY_600)
        pdf.cell(ew - 8, 8, "Focus: " + focus)
        pdf.ln(9)

    # Actions
    pdf.ln(2)
    for i, action in enumerate(actions, 1):
        _numbered(pdf, i, action, num_color=theme["num"])

    pdf.ln(5)


def _chips(pdf, items, bg_color, text_color=WHITE):
    """Render items as inline pill-shaped chips."""
    x = _lm(pdf)
    max_x = pdf.w - pdf.r_margin
    chip_h = 9
    gap = 3

    y = pdf.get_y()
    for item in items:
        if not item:
            continue
        label = _s(str(item))
        pdf.set_font("Helvetica", "", 8.5)
        tw = pdf.get_string_width(label) + 10

        if x + tw > max_x and x > _lm(pdf):
            x = _lm(pdf)
            y = pdf.get_y() + chip_h + gap
            pdf.set_y(y)

        # Pill background
        pdf.set_fill_color(*bg_color)
        radius = chip_h / 2
        pdf.rect(x + radius, y, tw - 2 * radius, chip_h, "F")
        pdf.ellipse(x,                y, chip_h, chip_h, "F")
        pdf.ellipse(x + tw - chip_h, y, chip_h, chip_h, "F")

        # Label
        pdf.set_xy(x, y)
        pdf.set_text_color(*text_color)
        pdf.cell(tw, chip_h, label, align="C")

        x += tw + gap

    pdf.set_y(pdf.get_y() + chip_h + 4)
    pdf.set_x(_lm(pdf))


def _info_box(pdf, text, bg, border_color):
    """Shaded info box."""
    x = _lm(pdf)
    ew = _ew(pdf)
    y = pdf.get_y()

    # Measure height by temporarily rendering
    lines = pdf.multi_cell(ew - 12, 5.5, _s(text), split_only=True)
    box_h = len(lines) * 5.5 + 8

    # Draw background
    pdf.set_fill_color(*bg)
    pdf.rect(x, y, ew, box_h, "F")

    # Left border
    pdf.set_fill_color(*border_color)
    pdf.rect(x, y, 4, box_h, "F")

    # Text
    pdf.set_xy(x + 8, y + 4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY_800)
    pdf.multi_cell(ew - 12, 5.5, _s(text))
    pdf.set_y(y + box_h + 4)
    pdf.set_x(x)


def _two_col_list(pdf, items, color):
    """Render items in two columns."""
    x = _lm(pdf)
    ew = _ew(pdf)
    col_w = ew / 2 - 4
    y = pdf.get_y()

    for i, item in enumerate(items):
        if not item:
            continue
        col = i % 2
        row = i // 2
        cx = x + col * (col_w + 8)
        cy = y + row * 7

        pdf.set_xy(cx, cy)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*color)
        pdf.cell(5, 6, "->")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GRAY_800)
        pdf.multi_cell(col_w - 5, 6, _s(str(item)))

    rows = (len(items) + 1) // 2
    pdf.set_y(y + rows * 7 + 4)
    pdf.set_x(x)


# ─── Main Entry ─────────────────────────────────────────────────────────────

def generate_roadmap_pdf(student_name: str, roadmap: dict) -> bytes:
    pdf = RoadmapPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    today_str    = date.today().strftime("%B %d, %Y")
    field        = roadmap.get("detected_field", "Engineering")
    summary      = roadmap.get("summary", "")
    prob_ctx     = roadmap.get("probability_context", "")
    phases       = roadmap.get("phases", []) or []
    quick_wins   = roadmap.get("quick_wins", []) or []
    companies    = roadmap.get("companies_to_target", []) or []
    skills       = roadmap.get("skills_to_learn", []) or []
    certs        = roadmap.get("certifications", []) or []
    projects     = roadmap.get("project_ideas", []) or []
    interview    = roadmap.get("interview_prep", []) or []

    # ── Header ──────────────────────────────────────────────────────────────
    _draw_header(pdf, student_name, field, today_str)

    # ── Assessment ──────────────────────────────────────────────────────────
    _section_title(pdf, "Placement Assessment", "A", BLUE)

    if summary:
        _info_box(pdf, summary, (239, 246, 255), BLUE)

    if prob_ctx:
        _info_box(pdf, prob_ctx, (245, 243, 255), PURPLE)

    pdf.ln(2)

    # ── 6-Month Plan ────────────────────────────────────────────────────────
    if phases:
        _section_title(pdf, "Your 6-Month Action Plan", "P", PURPLE)
        for i, phase in enumerate(phases[:3]):
            _phase_card(pdf, phase, i)

    # ── Quick Wins ──────────────────────────────────────────────────────────
    if quick_wins:
        _section_title(pdf, "Quick Wins This Week", "Q", TEAL)
        for item in quick_wins:
            _bullet(pdf, item, symbol=">>")
        pdf.ln(3)

    # ── Companies ───────────────────────────────────────────────────────────
    if companies:
        _section_title(pdf, "Companies to Target", "C", AMBER)
        _chips(pdf, companies, AMBER)

    # ── Skills ──────────────────────────────────────────────────────────────
    if skills:
        _section_title(pdf, "Skills to Build", "S", BLUE)
        _two_col_list(pdf, skills, BLUE)

    # ── Certifications ──────────────────────────────────────────────────────
    if certs:
        _section_title(pdf, "Recommended Certifications", "C", PURPLE)
        for item in certs:
            _bullet(pdf, item, symbol="*")
        pdf.ln(2)

    # ── Projects ────────────────────────────────────────────────────────────
    if projects:
        _section_title(pdf, "Project Ideas to Build", "P", GREEN)
        for item in projects:
            _bullet(pdf, item, color=GRAY_800)
        pdf.ln(2)

    # ── Interview Prep ──────────────────────────────────────────────────────
    if interview:
        _section_title(pdf, "Interview Preparation Areas", "I", RED)
        _two_col_list(pdf, interview, RED)

    # Output
    result = pdf.output()
    return bytes(result) if isinstance(result, (bytes, bytearray)) else result.encode("latin-1")