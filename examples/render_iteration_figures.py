"""Render slide-ready comparison figures and an offline visual index from saved results."""

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


INK, MUTED, PAPER, LINE = "#20362e", "#59675f", "#fbfaf6", "#d8dfd7"
TEAL, AMBER = "#21685c", "#9b492c"
SHORT_NAMES = {
    "object_green_cap": "Green cap", "object_blue_bike": "Blue bicycle",
    "object_sketchbook": "Sketchbook", "object_cast_iron_skillet": "Skillet",
    "object_blue_mug": "Blue mug",
}
LABELS = {
    "original_first_pass": ("Original pipeline · pass 1", "Fresh control · shared starting image"),
    "original_second_pass": ("Original pipeline · pass 2", "Fresh control · original edit policy"),
    "modified_second_pass": ("Modified pipeline · pass 2", "Maya portrait supplied again"),
}


def font(size, bold=False):
    names = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
             if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError("Install Arial or DejaVu Sans to render readable figures.")


def number(value):
    return "—" if value is None else f"{value:.4f}"


def asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or out-of-package image: " + relative)
    return path


def wrapped(draw, text, position, width, size=30, fill=INK, bold=False, line_height=None):
    """Draw wrapped text without truncating its contents; return its bottom edge."""
    x, y = position
    face, lines = font(size, bold), []
    for paragraph in str(text).splitlines():
        current = ""
        for word in paragraph.split():
            trial = (current + " " + word).strip()
            if current and draw.textlength(trial, font=face) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    step = line_height or int(size * 1.3)
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += step
    return y


def photo(canvas, path, rectangle):
    x, y, width, height = rectangle
    image = ImageOps.contain(Image.open(path).convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width-image.width)//2, y + (height-image.height)//2))


def maya(column):
    return next(row for row in column["face_scores"] if row["person_id"] == "person_maya")


def figure(root, scene, output):
    canvas = Image.new("RGB", (2400, 1350), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((78, 40), "CIS 5810  /  PORTRAIT-REFERENCE ITERATION", font=font(27, True), fill=TEAL)
    wrapped(draw, f"{scene['name']} · pass-by-pass comparison", (78, 90), 2244, 58, bold=True)
    columns = scene["columns"]
    if len(columns) == 1:
        column = columns[0]
        draw.text((78, 180), "One real pass — no second-pass comparison", font=font(35), fill=MUTED)
        photo(canvas, asset(root, column["image"]), (90, 270, 850, 850))
        draw.text((1060, 280), "Shared single-pass control", font=font(47, True), fill=TEAL)
        y = wrapped(draw, "The original and modified runs reuse this exact image. No portrait-retention edit was applied.",
                    (1060, 350), 1200, 37, fill=MUTED)
        y += 50
        draw.text((1060, y), "FACE SIMILARITY · ARCFACE", font=font(28, True), fill=TEAL)
        for row in column["face_scores"]:
            y += 62
            draw.text((1060, y), row["name"], font=font(36), fill=INK)
            draw.text((2200, y), number(row["similarity"]), font=font(40, True), fill=INK, anchor="ra")
        y += 100
        draw.text((1060, y), "OBJECT SIMILARITY · DINOV2 / CANDIDATES", font=font(28, True), fill=TEAL)
        for row in column["object_scores"]:
            y += 62
            draw.text((1060, y), SHORT_NAMES.get(row["object_id"], row["name"]), font=font(36), fill=INK)
            value = f"{number(row['similarity'])}   /   {row['candidate_count']}"
            draw.text((2200, y), value, font=font(40, True), fill=INK, anchor="ra")
    else:
        if len(columns) != 3:
            raise ValueError("Primary figure must contain three real passes, or one shared control.")
        for index, column in enumerate(columns):
            x = 78 + 758 * index
            label, context = LABELS.get(column["key"], (column["label"], ""))
            color = TEAL if index == 2 else INK
            draw.text((x, 181), label, font=font(34, True), fill=color)
            draw.text((x, 224), context, font=font(25), fill=MUTED)
            photo(canvas, asset(root, column["image"]), (x + 20, 272, 688, 688))
            row = maya(column)
            score_color = AMBER if scene.get("warning") and index == 2 else INK
            draw.text((x, 984), "Maya · ArcFace", font=font(32, True), fill=INK)
            draw.text((x+728, 976), number(row["similarity"]), font=font(45, True), fill=score_color, anchor="ra")
            draw.text((x, 1032), "DINOv2 object similarity  /  candidates", font=font(25), fill=MUTED)
            for offset, obj in enumerate(column["object_scores"]):
                y = 1071 + 43 * offset
                draw.text((x, y), SHORT_NAMES.get(obj["object_id"], obj["name"]), font=font(31), fill=INK)
                value = f"{number(obj['similarity'])}   /   {obj['candidate_count']}"
                draw.text((x+728, y), value, font=font(31, True), fill=INK, anchor="ra")
    draw.line((78, 1225, 2322, 1225), fill=LINE, width=2)
    message = scene.get("warning") or scene.get("note") or "Same saved story, reference assets, model and seeds. The first pass is identical in both fresh runs."
    bottom = wrapped(draw, message, (78, 1242), 2244, 28, fill=AMBER if scene.get("warning") else MUTED, bold=bool(scene.get("warning")))
    if bottom > 1318:
        raise ValueError(f"Figure note is too long for {scene['name']}; shorten the presentation note.")
    draw.text((78, 1314), "Cosine similarities, not accuracy percentages. — means no candidate. Objects use final-scene targets; pass 2 may intentionally introduce them. Placement and ownership are not measured.", font=font(23), fill=MUTED)
    canvas.save(output, optimize=True)


def render(root):
    data = json.loads((root / "results/comparison_data.json").read_text())
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    esc = html.escape
    generated = []
    for scene in data["scenes"]:
        relative = f"figures/{scene['photo_index']+1:02d}_{scene['month']:02d}_comparison.png"
        figure(root, scene, root / relative)
        generated.append(relative)

    def image(relative, label):
        asset(root, relative)
        return f'<a class="photo-link" href="{quote(relative)}"><img src="{quote(relative)}" alt="{esc(label)}" loading="lazy"></a>'

    def score_table(scene, kind):
        field, id_key = ("face_scores", "person_id") if kind == "faces" else ("object_scores", "object_id")
        ids = list(dict.fromkeys(row[id_key] for col in scene["columns"] for row in col[field]))
        tables = [{row[id_key]: row for row in col[field]} for col in scene["columns"]]
        title = "Face similarity · ArcFace" if kind == "faces" else "Object similarity · DINOv2"
        body = []
        for key in ids:
            label = next(rows[key]["name"] for rows in tables if key in rows)
            cells = []
            for rows in tables:
                row = rows.get(key)
                if not row:
                    cells.append('<td>Not scored</td>')
                else:
                    detail = (f'face candidate {row["face_index"]}' if row["face_index"] is not None else 'no assigned face') if kind == "faces" else f'{row["candidate_count"]} candidate(s)'
                    cells.append(f'<td><strong>{number(row["similarity"])}</strong><small>{esc(detail)}</small></td>')
            body.append(f'<tr><th scope="row">{esc(label)}</th>{"".join(cells)}</tr>')
        heads = ''.join(f'<th scope="col">{esc(col["label"])}</th>' for col in scene["columns"])
        return f'<h3>{title}</h3><div class="table-scroll"><table><thead><tr><th scope="col">Target</th>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

    def scene_html(scene, primary=True):
        columns = scene["columns"]
        cards = []
        for col in columns:
            row = maya(col)
            cards.append(f'<article class="photo-card"><h3>{esc(col["label"])}</h3>{image(col["image"], scene["name"]+" — "+col["label"])}<p class="maya-score">Maya · ArcFace <strong>{number(row["similarity"])}</strong></p></article>')
        warning = f'<p class="warning"><strong>Interpretation warning.</strong> {esc(scene["warning"])}</p>' if scene.get("warning") else ''
        note = f'<p class="note">{esc(scene["note"])}</p>' if scene.get("note") else ''
        links = ''
        if primary:
            relative = generated[data["scenes"].index(scene)]
            links = f'<p><a class="download" href="{quote(relative)}">Open slide-ready PNG ↗</a></p>'
        return f'<section id="{"scene" if primary else "historical"}-{scene["month"]}"><header class="scene-heading"><h2>{esc(scene["name"])}</h2>{links}</header>{warning}{note}<div class="photos cols-{len(columns)}">{"".join(cards)}</div>{score_table(scene,"objects")}<details><summary>All character scores and assigned faces</summary>{score_table(scene,"faces")}</details></section>'

    nav = ''.join(f'<a href="#scene-{s["month"]}">{esc(s["name"])}</a>' for s in data["scenes"])
    content = [f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(data['title'])}</title>
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:{PAPER};color:{INK};font:17px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:48px 32px}}h1{{font-size:clamp(32px,4vw,56px);line-height:1.1;max-width:1100px;margin:12px 0 20px}}h2{{font-size:30px;margin:0}}h3{{font-size:18px;line-height:1.3}}p{{max-width:1100px}}.eyebrow{{color:{TEAL};font-weight:700;letter-spacing:.09em;font-size:13px}}a{{color:{TEAL}}}.intro{{font-size:21px;color:{MUTED}}}.notice,.warning{{padding:18px 22px;background:#f5e7dd;border-left:4px solid {AMBER}}}.note{{color:{MUTED}}}.links,nav{{display:flex;gap:14px 24px;flex-wrap:wrap;margin:22px 0}}nav{{position:sticky;top:0;z-index:2;background:{PAPER};padding:15px 0;border-block:1px solid {LINE}}}nav a{{text-decoration:none;font-weight:650}}section{{padding:32px 0 44px;border-bottom:1px solid {LINE};scroll-margin-top:75px}}.scene-heading{{display:flex;align-items:center;justify-content:space-between;gap:18px}}.photos{{display:grid;gap:22px;align-items:start}}.cols-3{{grid-template-columns:repeat(3,minmax(0,1fr))}}.cols-2{{grid-template-columns:repeat(2,minmax(0,1fr))}}.cols-1{{max-width:650px;margin:20px auto}}.photo-card h3{{min-height:48px}}.photo-link{{display:block}}.photo-link img{{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#e9ebe5}}.maya-score{{display:flex;justify-content:space-between;gap:16px;margin:12px 0 24px}}.maya-score strong{{font-size:25px}}table{{width:100%;border-collapse:collapse;font-size:15px;font-variant-numeric:tabular-nums}}th,td{{padding:13px 14px;border-bottom:1px solid {LINE};text-align:left;vertical-align:top}}thead{{background:#e9eee7}}tbody th{{font-weight:550}}td small{{display:block;color:{MUTED};font-size:12px}}.table-scroll{{overflow:auto}}details{{margin:22px 0}}summary{{cursor:pointer;font-weight:650}}.download{{font-size:15px}}.history{{background:#f0f1eb;padding:24px}}footer{{color:{MUTED};font-size:14px;padding-top:30px}}@media(max-width:850px){{main{{padding:25px 18px}}.photos.cols-3{{gap:10px}}.photo-card h3{{font-size:14px;min-height:55px}}.maya-score{{font-size:12px;display:block}}.maya-score strong{{display:block;font-size:21px}}.scene-heading{{align-items:start}}h2{{font-size:25px}}th,td{{padding:10px 8px}}}}@media(max-width:520px){{.photos.cols-3,.photos.cols-2{{grid-template-columns:1fr}}.photo-card h3{{min-height:0;font-size:18px}}.maya-score{{display:flex;font-size:15px}}nav{{font-size:13px;gap:12px}}}}@media print{{nav,.links{{display:none}}section{{break-before:page}}details{{display:block}}}}
</style></head><body><main><p class="eyebrow">CIS 5810 · DOCUMENTED PIPELINE ITERATION</p><h1>{esc(data['title'])}</h1><p class="intro">{esc(data['subtitle'])}</p>
<p>Each edited scene begins with the exact same first-pass image. Compare the original edit policy with the modified edit that resupplies Maya’s portrait. “Original pipeline” here means the fresh paired control; the older historical album is preserved separately below.</p>
<p class="notice"><strong>Reading the numbers:</strong> ArcFace and DINOv2 are separate cosine-similarity measures, not accuracy percentages. A higher score does not prove the correct person, object placement or ownership. Missing detections remain missing. Multiple detector candidates and background inside crops can affect object scores. Object targets come from the final scene, so missing first-pass objects may be intentionally introduced in pass 2.</p>
<div class="links"><a href="ANALYSIS.md">Analysis &amp; presentation notes</a><a href="results/face_scores.csv">Face results CSV</a><a href="results/object_scores.csv">Object results CSV</a><a href="results/comparison_data.json">Data behind this view</a></div><nav aria-label="Scenes">{nav}<a href="#historical">Historical originals</a></nav>''']
    content.extend(scene_html(s) for s in data["scenes"])
    if data.get("historical_scenes"):
        content.append('<section id="historical"><h2>Historical originals — preserved</h2><p>These are the earlier saved images. They are context, not the matched starting point for the new experiment: rerunning the provider with the same seeds did not reproduce them exactly.</p><details class="history"><summary>Open the historical first- and second-pass results</summary>')
        content.extend(scene_html(s, False) for s in data["historical_scenes"])
        content.append('</details></section>')
    content.append('<footer>All images and recorded scores are saved in this package. This page works offline and makes no API calls. The experiment reports mixed findings; it does not claim all edited photos improved.</footer></main></body></html>')
    (root / "index.html").write_text('\n'.join(content), encoding="utf-8")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    print(json.dumps({"figures": render(args.package.resolve())}, indent=2))
