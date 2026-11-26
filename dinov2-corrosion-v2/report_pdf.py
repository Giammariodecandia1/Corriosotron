"""One-page A4 visual inspection report for the analysed ROI.

Layout follows the reference "Report Ispezione Corrosione.pdf": header card with
status pill, four-column info strip, two image panels, summary card, operator
notes card and a footer with the AI disclaimer pill and company name.
Measurements below are distances from the top of the page, in points.
"""
import io
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

PW, PH = A4
L, R = 45, PW - 45
W = R - L
NAVY, BLUE, PALE = HexColor("#102B43"), HexColor("#147EAE"), HexColor("#EEF6FA")
INK, MUTED, RULE = HexColor("#193244"), HexColor("#587080"), HexColor("#D6E3E9")
CYAN, LIGHT, WHITE = HexColor("#3BBBE8"), HexColor("#C7DDE9"), HexColor("#FFFFFF")
STRIP, PILL_BG, PILL_LINE = HexColor("#F7FAFC"), HexColor("#1B3550"), HexColor("#4A6378")
SEPARATOR = HexColor("#2A4A62")
GREEN, GREEN_DOT = HexColor("#1F7A4D"), HexColor("#22A060")
GREEN_BG, GREEN_LINE = HexColor("#E9F8EF"), HexColor("#B8E6CB")
AMBER, RED = HexColor("#F9CA3D"), HexColor("#EE3E58")


def T(top):
    """PDF y coordinate for a distance from the top of the page."""
    return PH - top


def line(c, value, x, y, width, size=9, bold=False, color=INK, mono=False):
    font = "Courier" if mono else ("Helvetica-Bold" if bold else "Helvetica")
    value = str(value or "")
    while value and c.stringWidth(value, font, size) > width:
        value = value[:-2]
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, value)


def pair(c, label, value, x, y, size=8, width=200):
    """Muted label followed by a bold value on the same baseline."""
    c.setFont("Helvetica", size)
    c.setFillColor(LIGHT)
    c.drawString(x, y, label)
    line(c, value, x + c.stringWidth(label, "Helvetica", size), y, width, size, True, WHITE)


def pct(value):
    return "n/d" if value is None else f"{value:.1f}%".replace(".", ",")


def image_data(image):
    image = image.convert("RGB")
    image.thumbnail((1400, 1400))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=88)
    data.seek(0)
    return data


def header(c, title, subtitle):
    """Header card with left accent bar and status pill (top 45-110)."""
    c.setFillColor(NAVY)
    c.roundRect(L, T(110), W, 65, 6, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(L, T(110), 4, 65, fill=1, stroke=0)
    line(c, "EAGLE PROJECTS • RUSTRON / CORROSION INTELLIGENCE", L + 18, T(66), 330, 8, True, CYAN)
    line(c, title, L + 18, T(85), 330, 19, True, WHITE)
    line(c, subtitle, L + 18, T(99), 330, 9, False, LIGHT)
    pill_w = 142
    pill_x = R - 12 - pill_w
    c.setFillColor(PILL_BG)
    c.setStrokeColor(PILL_LINE)
    c.roundRect(pill_x, T(88), pill_w, 21, 10.5, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(HexColor("#E6F1F7"))
    c.drawCentredString(pill_x + pill_w / 2, T(77.5) - 2.6, "ISPEZIONE STRUTTURALE")


def footer(c, page_label=""):
    """Rule, AI pill, optional page label and company name (top 782-812)."""
    c.setStrokeColor(RULE)
    c.line(L, T(782), R, T(782))
    pill_text = "Rustron / Corrosion Intelligence · Analisi AI visiva da modelli interni."
    footer_w = c.stringWidth(pill_text, "Helvetica-Bold", 8) + 30
    c.setFillColor(GREEN_BG)
    c.setStrokeColor(GREEN_LINE)
    c.roundRect(L, T(812), footer_w, 18, 9, fill=1, stroke=1)
    c.setFillColor(GREEN_DOT)
    c.circle(L + 10, T(803), 2.5, fill=1, stroke=0)
    line(c, pill_text, L + 18, T(806), footer_w - 24, 8, True, GREEN)
    c.setFillColor(MUTED)
    if page_label:
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(PW / 2 + 60, T(806), page_label)
    c.setFont("Helvetica", 8.5)
    c.drawRightString(R, T(806), "Eagle Projects S.p.A.")


BAND_COLORS = {"rilevante": RED, "da monitorare": HexColor("#D79A00")}
BAND_LABELS = {"rilevante": "Rilevante", "da monitorare": "Da monitorare"}
MAX_TABLE_ROWS = 30


def anomalies_page(c, photo_name, classified_image, clusters):
    """Page 2: classification map with numbered anomaly boxes and the anomaly table."""
    found = clusters.get("clusters") or []
    header(c, "Anomalie numerate", f"{len(found)} sopra soglia  ·  {photo_name or ''}")
    iw, ih = classified_image.size
    top = 125
    if iw / ih > 0.9:   # wide map: full width on top, table below
        bx, btop, bw, bh = L, top, W, 330
        table_x, table_top, table_w = L, top + bh + 16, W
    else:               # tall map (typical tower crop): map left, table right
        bx, btop, bw, bh = L, top, 250, 645
        table_x, table_top, table_w = L + 264, top, W - 264

    c.setFillColor(PALE)
    c.roundRect(bx, T(btop + bh), bw, bh, 6, fill=1, stroke=0)
    scale = min((bw - 12) / iw, (bh - 12) / ih)
    dw, dh = iw * scale, ih * scale
    ox, oy = bx + (bw - dw) / 2, btop + (bh - dh) / 2   # image left, image top
    c.drawImage(ImageReader(image_data(classified_image)), ox, T(oy + dh), dw, dh)
    for cl in reversed(found):   # A1 drawn last, on top
        x0, y0, x1, y1 = cl["bbox_px"]
        rx, ry = ox + x0 * scale, oy + y0 * scale
        rw, rh = max((x1 - x0) * scale, 4), max((y1 - y0) * scale, 4)
        color = BAND_COLORS[cl["band"]]
        for width, stroke in ((2.6, WHITE), (1.2, color)):
            c.setLineWidth(width)
            c.setStrokeColor(stroke)
            c.rect(rx, T(ry + rh), rw, rh, fill=0, stroke=1)
        c.setFont("Helvetica-Bold", 6.5)
        tw = c.stringWidth(cl["id"], "Helvetica-Bold", 6.5) + 5
        tag_top = ry - 9 if ry - 9 >= oy else ry
        c.setFillColor(color)
        c.rect(rx, T(tag_top + 9), tw, 9, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.drawString(rx + 2.5, T(tag_top + 6.6), cl["id"])
    c.setLineWidth(1)

    # Table: ID, band, % of visible metal, % intense, extent on tower height.
    base = (26, 66, 48, 48, 53)
    widths = [w * table_w / sum(base) for w in base]
    heads = ("ID", "FASCIA", "% METALLO", "% INTENSA", "ESTENSIONE")
    c.setFillColor(PALE)
    c.rect(table_x, T(table_top + 16), table_w, 16, fill=1, stroke=0)
    x = table_x
    for head, width in zip(heads, widths):
        line(c, head, x + 4, T(table_top + 11), width - 6, 6.6, True, NAVY)
        x += width
    rows_fit = int((770 - table_top - 16 - 44) // 14)
    shown = found[:max(0, min(MAX_TABLE_ROWS, rows_fit))]
    y = table_top + 16
    for cl in shown:
        cells = (cl["id"], None, pct(cl["pct_of_visible_metal"]), pct(cl["intense_pct_of_anomaly"]),
                 pct(cl["extent_pct_of_tower_height"]) + " alt.")
        x = table_x
        for n, (value, width) in enumerate(zip(cells, widths)):
            if n == 1:
                label_text = BAND_LABELS[cl["band"]]
                pw = c.stringWidth(label_text, "Helvetica-Bold", 6.3) + 8
                c.setFillColor(BAND_COLORS[cl["band"]])
                c.roundRect(x + 3, T(y + 11.5), pw, 9, 4.5, fill=1, stroke=0)
                line(c, label_text, x + 7, T(y + 9.3), width - 8, 6.3, True, WHITE)
            else:
                line(c, value, x + 4, T(y + 10), width - 6, 7.5, n == 0, INK)
            x += width
        y += 14
        c.setStrokeColor(RULE)
        c.line(table_x, T(y), table_x + table_w, T(y))
    if not found:
        line(c, "Nessuna anomalia sopra soglia.", table_x + 4, T(y + 14), table_w - 8, 8.5, True, INK)
        y += 18
    if len(found) > len(shown):
        line(c, f"Altre {len(found) - len(shown)} anomalie sopra soglia nel JSON / CSV.",
             table_x + 4, T(y + 13), table_w - 8, 7.5, True, INK)
        y += 14

    thresholds = clusters.get("thresholds_pct_of_visible_metal") or {}
    legend = (f"Fasce: Rilevante almeno {pct(thresholds.get('rilevante'))} del metallo visibile; "
              f"Da monitorare almeno {pct(thresholds.get('da monitorare'))}. "
              f"Zone più piccole senza ID: {clusters.get('below_threshold_count', 0)}. "
              f"Frammenti uniti entro {clusters.get('merge_distance_px') or 0} px "
              f"(1% dell'estensione del traliccio). "
              "Estensione = altezza dell'anomalia in % dell'altezza del traliccio inquadrato. "
              "ID numerati per grandezza: A1 è la più estesa.")
    for n, part in enumerate(simpleSplit(legend, "Helvetica", 7, table_w - 8)[:6]):
        line(c, part, table_x + 4, T(y + 16 + 10 * n), table_w - 8, 7, color=MUTED)


def build_pdf(photo_name, roi_image, classified_image, report, operator_notes="", clusters=None):
    """Return the PDF. Page 1 always; page 2 (numbered anomalies) only when `clusters`
    is given, i.e. the operator ticked "Clusterizza anomalie". Date and GPS are read
    from the photo metadata (report["shot_captured_at"], report["shot_coordinates"])."""
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    c.setTitle("Rustron / Corrosion Intelligence - ispezione corrosione")
    c.setAuthor("Eagle Projects")
    header(c, "Analisi visiva della corrosione", "Sostegni metallici AT / AAT")

    # Information strip: photo, coordinates, profile, date/time.
    c.setFillColor(STRIP)
    c.setStrokeColor(RULE)
    c.roundRect(L, T(172), W, 50, 5, fill=1, stroke=1)
    line(c, "FOTOGRAFIA", L + 12, T(134), 120, 7, True, MUTED)
    line(c, photo_name or "Non indicata", L + 12, T(157), 228, 7.2, color=BLUE, mono=True)
    line(c, "COORDINATE", L + 252, T(134), 80, 7, True, MUTED)
    lat_text, _, lon_text = (report.get("shot_coordinates") or "").partition(", ")
    line(c, lat_text, L + 252, T(157), 86, 8, True)
    line(c, lon_text, L + 252, T(167), 86, 8, True)
    line(c, "PROFILO", L + 338, T(134), 80, 7, True, MUTED)
    line(c, report.get("profile_name") or "Non indicato", L + 338, T(157), 80, 8.5, True)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawRightString(R - 12, T(134), "DATA / ORA")
    c.setFont("Helvetica", 8)
    c.drawRightString(R - 12, T(157), report.get("shot_captured_at") or "")

    # Image panels: analysed ROI and classification map.
    gap, panel_top, panel_h, head_h = 12, 184, 382, 23
    box_w = (W - gap) / 2
    panels = ((roi_image, "FOTOGRAFIA / AREA ANALIZZATA"),
              (classified_image, "MAPPA DI CLASSIFICAZIONE"))
    # Legend of the excluded zones (automatic = model, manual = drawn by the operator), only when present.
    auto_px = (report.get("object_recognition") or {}).get("excluded_pixels_in_roi") or 0
    manual = report.get("insulator_exclusions") or {}
    manual_n = len(manual.get("regions") or [])
    legend = []
    if auto_px:
        legend.append((HexColor("#547EFF"), "Esclusi dal modello (YOLO)"))
    if manual_n:
        legend.append((HexColor("#28BEFF"), f"Esclusi a mano ({manual_n})"))
    for index, (image, title) in enumerate(panels):
        x = L + index * (box_w + gap)
        c.setFillColor(PALE)
        c.roundRect(x, T(panel_top + panel_h), box_w, panel_h, 6, fill=1, stroke=0)
        line(c, title, x + 9, T(panel_top + 15), box_w - 18, 7.5, True, NAVY)
        area_top = panel_top + head_h + 6
        area_h = panel_h - head_h - 12 - (14 if legend and index == 1 else 0)
        if legend and index == 1:
            lx = x + 9
            for color, text in legend:
                c.setFillColor(color)
                c.rect(lx, T(panel_top + panel_h - 9), 8, 8, fill=1, stroke=0)
                line(c, text, lx + 11, T(panel_top + panel_h - 7.5), 120, 6.8, False, INK)
                lx += 11 + c.stringWidth(text, "Helvetica", 6.8) + 10
        iw, ih = image.size
        scale = min((box_w - 16) / iw, area_h / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(ImageReader(image_data(image)), x + (box_w - dw) / 2,
                    T(area_top + area_h) + (area_h - dh) / 2, dw, dh)

    # Summary card: corrosion share of the visible metal and its classes.
    top = 577
    classes = report.get("class_pct_of_visible_metal") or {}
    calibrated = set(report.get("calibrated_classes") or ())
    c.setFillColor(NAVY)
    c.roundRect(L, T(top + 118), W, 118, 6, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(L, T(top + 3), W, 3, fill=1, stroke=0)
    line(c, "METALLO VISIBILE CLASSIFICATO CON RUGGINE", L + 18, T(top + 22), W - 36, 8, True, LIGHT)
    line(c, pct(report.get("corrosion_pct_of_visible_metal")), L + 18, T(top + 58), 230, 30, True, WHITE)
    line(c, "Totale superficie affetta da ruggine", L + 18, T(top + 77), 230, 8.5, color=LIGHT)
    for x_offset, title, key, color in ((376, "MEDIA", "medium", AMBER),
                                        (445, "INTENSA", "intense", RED)):
        x = L + x_offset
        c.setFillColor(color)
        c.circle(x, T(top + 40), 3.5, fill=1, stroke=0)
        line(c, title, x + 9, T(top + 44), 90, 8, True, LIGHT)
        if key in calibrated:
            line(c, pct(classes.get(key)), x, T(top + 67), 130, 15, True, WHITE)
        else:
            line(c, "Non calibrata", x, T(top + 67), 130, 9, True, WHITE)
    c.setStrokeColor(SEPARATOR)
    c.line(L + 18, T(top + 90), R - 18, T(top + 90))
    pair(c, "Zone da verificare: ", pct(classes.get("uncertain")), L + 18, T(top + 107))
    scope_label, scope_value = "Ambito: ", "metallo visibile nella foto / ROI"
    objects = (report.get("object_recognition") or {}).get("summary") or []
    if objects:  # automatic object recognition (v2): say what was removed or kept
        words = {"exclude": "esclusi", "focus": "solo"}
        parts = [f'{words[o["mode"]]} {o["count_in_roi"]} {o["class_name"]}' for o in objects if o["mode"] in words]
        scope_value = "metallo visibile · " + ", ".join(parts[:3])
    if manual_n:
        scope_value += f" · {manual_n} zone escluse a mano"
    c.setFont("Helvetica", 8)
    while c.stringWidth(scope_value, "Helvetica-Bold", 8) > 320:
        scope_value = scope_value[:-2]
    scope_w = c.stringWidth(scope_label, "Helvetica", 8) + c.stringWidth(scope_value, "Helvetica-Bold", 8)
    pair(c, scope_label, scope_value, R - 18 - scope_w, T(top + 107), width=330)

    # Operator notes card.
    notes_top = 705
    c.setStrokeColor(RULE)
    c.roundRect(L, T(notes_top + 58), W, 58, 6, fill=0, stroke=1)
    line(c, "NOTE DELL'OPERATORE", L + 12, T(notes_top + 16), W - 24, 7.5, True, NAVY)
    note_lines = simpleSplit((operator_notes or "").strip(), "Helvetica", 8.5, W - 24)[:2]
    for n, note in enumerate(note_lines):
        line(c, note, L + 12, T(notes_top + 31 + 14 * n), W - 24, 8.5)
    if not note_lines:
        c.setStrokeColor(RULE)
        c.line(L + 12, T(notes_top + 35), R - 12, T(notes_top + 35))
        c.line(L + 12, T(notes_top + 50), R - 12, T(notes_top + 50))

    footer(c, "Pag. 1/2" if clusters else "")
    c.showPage()
    if clusters:
        anomalies_page(c, photo_name, classified_image, clusters)
        footer(c, "Pag. 2/2")
        c.showPage()
    c.save()
    return out.getvalue()
