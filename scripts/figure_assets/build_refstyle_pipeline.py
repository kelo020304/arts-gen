#!/usr/bin/env python3
"""Draw a deterministic PhyxForge-style PartSeg pipeline overview."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/code/arts-gen")
ASSET = ROOT / "figure_assets" / "01816801a27444cbb5cfb934de39d483"
OUT = ROOT / "figure_out"

W, H = 3200, 1300
WHITE = (255, 255, 255)
INK = (45, 48, 55)
LINE = (74, 82, 94)
SOFT_LINE = (146, 153, 164)
PANEL = (252, 252, 250)
CREAM = (245, 239, 203)
LAV = (235, 230, 251)
MINT = (226, 244, 232)
SAGE = (232, 241, 229)
ORANGE = (249, 226, 198)
PINK = (252, 229, 236)
BLUE = (73, 139, 205)
GREEN = (91, 168, 111)
AMBER = (226, 143, 65)
RED = (214, 56, 52)


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        names = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        names = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for name in names:
        p = Path(name)
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F20 = font(20, bold=True)
F22 = font(22, bold=True)
F26 = font(26, bold=True)
F30 = font(30, bold=True, serif=True)
F36 = font(36, bold=True, serif=True)
F44 = font(44, bold=True, serif=True)


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill=INK, fnt=F18, spacing: int = 5) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = (box[1] + box[3] - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text(((box[0] + box[2] - tw) / 2, y), line, fill=fill, font=fnt)
        y += th + spacing


def dashed_round(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, outline=SOFT_LINE, width: int = 3, dash: int = 14, gap: int = 9, fill=None) -> None:
    if fill is not None:
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    x0, y0, x1, y1 = box
    # Draw a thin rounded rectangle first, then mask with white gaps for a clean dashed effect.
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=width)
    for x in range(x0 + radius, x1 - radius, dash + gap):
        draw.line((x + dash, y0, min(x + dash + gap, x1 - radius), y0), fill=WHITE, width=width + 2)
        draw.line((x + dash, y1, min(x + dash + gap, x1 - radius), y1), fill=WHITE, width=width + 2)
    for y in range(y0 + radius, y1 - radius, dash + gap):
        draw.line((x0, y + dash, x0, min(y + dash + gap, y1 - radius)), fill=WHITE, width=width + 2)
        draw.line((x1, y + dash, x1, min(y + dash + gap, y1 - radius)), fill=WHITE, width=width + 2)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str | None = None, *, fill=(249, 250, 252), outline=(157, 166, 178), width: int = 2, radius: int = 14, fnt=F18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    if label:
        text_center(draw, box, label, fnt=fnt)


def new_badge(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    box = (x, y, x + 58, y + 28)
    draw.rounded_rectangle(box, radius=7, fill=RED, outline=RED)
    text_center(draw, box, "NEW", fill=WHITE, fnt=F14)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, width: int = 4, fill=LINE, dashed: bool = False) -> None:
    x0, y0 = start
    x1, y1 = end
    if dashed:
        segments = max(8, int(math.hypot(x1 - x0, y1 - y0) // 18))
        for i in range(segments):
            if i % 2 == 0:
                a = i / segments
                b = (i + 1) / segments
                draw.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=fill, width=width)
    else:
        draw.line((x0, y0, x1, y1), fill=fill, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    length = 18
    left = (x1 - length * math.cos(ang - 0.45), y1 - length * math.sin(ang - 0.45))
    right = (x1 - length * math.cos(ang + 0.45), y1 - length * math.sin(ang + 0.45))
    draw.polygon([(x1, y1), left, right], fill=fill)


def elbow(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], *, width: int = 4, fill=LINE, dashed: bool = False) -> None:
    for a, b in zip(pts[:-1], pts[1:]):
        arrow(draw, a, b, width=width, fill=fill, dashed=dashed)


def trapezoid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, *, direction: str, fill=SAGE, outline=(104, 132, 104), fnt=F16) -> None:
    x0, y0, x1, y1 = box
    inset = max(24, (y1 - y0) // 5)
    if direction == "narrow_right":
        poly = [(x0, y0), (x1, y0 + inset), (x1, y1 - inset), (x0, y1)]
    elif direction == "wide_right":
        poly = [(x0, y0 + inset), (x1, y0), (x1, y1), (x0, y1 - inset)]
    else:
        poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    draw.polygon(poly, fill=fill, outline=outline)
    draw.line(poly + [poly[0]], fill=outline, width=2)
    text_center(draw, box, label, fnt=fnt)


def crop_nonwhite(im: Image.Image, threshold: int = 248, pad: int = 20) -> Image.Image:
    rgb = im.convert("RGB")
    pix = rgb.load()
    xs, ys = [], []
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = pix[x, y]
            if min(r, g, b) < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgb
    x0, x1 = max(0, min(xs) - pad), min(rgb.width, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(rgb.height, max(ys) + pad)
    return rgb.crop((x0, y0, x1, y1))


def paste_img(canvas: Image.Image, path: Path, box: tuple[int, int, int, int], *, crop: bool = True, border: bool = False) -> None:
    im = Image.open(path).convert("RGB")
    if crop:
        im = crop_nonwhite(im)
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    scale = min(max_w / im.width, max_h / im.height)
    resized = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
    px = x0 + (max_w - resized.width) // 2
    py = y0 + (max_h - resized.height) // 2
    canvas.paste(resized, (px, py))
    if border:
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((x0, y0, x1, y1), radius=10, outline=(184, 193, 205), width=2)


def tiny_tokens(draw: ImageDraw.ImageDraw, x: int, y: int, colors: list[tuple[int, int, int]], rows: int = 2, cols: int = 5, size: int = 28) -> None:
    for r in range(rows):
        for c in range(cols):
            color = colors[(r * cols + c) % len(colors)]
            draw.rounded_rectangle((x + c * (size + 8), y + r * (size + 8), x + c * (size + 8) + size, y + r * (size + 8) + size), radius=5, fill=(250, 252, 252), outline=color, width=2)


def draw_seg_trunk(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    for i in range(7):
        dx = i * 13
        draw.rounded_rectangle((x0 + dx, y0 + 10, x0 + dx + 44, y1 - 10), radius=8, fill=ORANGE, outline=(172, 108, 45), width=2)
        draw.line((x0 + dx + 10, y0 + 16, x0 + dx + 10, y1 - 16), fill=(192, 127, 64), width=2)
    text_center(draw, (x0 + 42, y0, x1, y1), "Seg\nTrunk\n🔥", fnt=F18)
    arrow(draw, (x0 + 80, y0 - 22), (x0 + 134, y0 - 22), width=3)
    arrow(draw, (x0 + 134, y0 - 22), (x0 + 80, y0 - 22), width=3)
    draw.text((x0 + 72, y0 - 58), "×K parts", fill=INK, font=F18)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Outer panels.
    dashed_round(draw, (35, 35, 1810, 1265), 26, fill=PANEL, outline=(105, 112, 123), width=3)
    dashed_round(draw, (1840, 35, 3165, 1265), 26, fill=PANEL, outline=(105, 112, 123), width=3)
    draw.text((80, 66), "(a) Pipeline Overview", fill=INK, font=F30)
    draw.text((1885, 66), "(b) Promptable Part Segmentation", fill=INK, font=F30)

    # VLM top bar.
    card(draw, (80, 115, 1760, 205), "Vision-Language Model / VLM Semantic Parser", fill=CREAM, outline=(145, 140, 95), width=2, radius=16, fnt=F36)

    # Left input and VLM outputs.
    card(draw, (80, 270, 420, 705), None, fill=(255, 255, 255), outline=(168, 175, 186), radius=15)
    draw.text((126, 288), "Multi-view renders", fill=INK, font=F22)
    paste_img(canvas, ASSET / "input_views.png", (108, 335, 392, 675), crop=False)
    arrow(draw, (250, 270), (250, 205), width=5, fill=LINE)
    draw.text((152, 225), "input to VLM", fill=LINE, font=F16)

    # SS generation path from VLM output.
    card(draw, (500, 260, 715, 420), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    draw.text((522, 278), "VLM output ①", fill=INK, font=F18)
    draw.text((522, 304), "selected 4-view group", fill=INK, font=F16)
    paste_img(canvas, ASSET / "input_views.png", (520, 332, 695, 405), crop=False)
    arrow(draw, (610, 205), (610, 260), width=4)

    trapezoid(draw, (755, 282, 905, 398), "DINOv2\n❄", direction="narrow_right", fill=(239, 243, 239), outline=(115, 126, 115), fnt=F18)
    card(draw, (940, 282, 1095, 398), "view\nembedding", fill=(250, 250, 250), outline=(156, 165, 176), radius=10, fnt=F16)
    tiny_tokens(draw, 962, 328, [BLUE, GREEN, AMBER], rows=1, cols=4, size=22)
    card(draw, (1135, 262, 1390, 420), "SS Flow Generator\n🔥\n4-view concat + embedding", fill=MINT, outline=(88, 146, 105), radius=14, fnt=F18)
    card(draw, (1455, 245, 1735, 445), None, fill=(255, 255, 255), outline=(169, 178, 190), radius=12)
    paste_img(canvas, ASSET / "voxel_whole.png", (1488, 270, 1705, 410))
    draw.text((1498, 416), "64³ occupancy voxel", fill=INK, font=F16)
    arrow(draw, (715, 340), (755, 340), width=4)
    arrow(draw, (905, 340), (940, 340), width=4)
    arrow(draw, (1095, 340), (1135, 340), width=4)
    arrow(draw, (1390, 340), (1455, 340), width=4)

    # VLM semantic names -> SAM3 -> masks.
    card(draw, (500, 565, 780, 760), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    draw.text((522, 585), "VLM output ②", fill=INK, font=F18)
    draw.text((522, 613), "per-part semantic names", fill=INK, font=F16)
    for i, (name, col) in enumerate([("part1: left handle", BLUE), ("part2: right handle", GREEN), ("part3: spout", AMBER)]):
        y = 652 + i * 30
        draw.rounded_rectangle((524, y - 4, 755, y + 24), radius=8, fill=(255, 255, 255), outline=col, width=2)
        draw.text((538, y), name, fill=col, font=F16)
    elbow(draw, [(470, 205), (470, 665), (500, 665)], width=4)

    card(draw, (845, 605, 1035, 725), "SAM3\nmask generator", fill=CREAM, outline=(140, 135, 95), radius=14, fnt=F18)
    card(draw, (1110, 525, 1480, 780), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    draw.text((1132, 545), "VLM output ③: 2D part masks", fill=INK, font=F18)
    paste_img(canvas, ASSET / "vlm_masks.png", (1140, 575, 1450, 735))
    arrow(draw, (780, 665), (845, 665), width=4)
    arrow(draw, (1035, 665), (1110, 665), width=4)

    card(draw, (1210, 850, 1585, 935), "Part prompts\nnames + SAM3 masks", fill=PINK, outline=(173, 94, 115), radius=14, fnt=F20)
    arrow(draw, (640, 760), (1210, 895), width=4)
    arrow(draw, (1295, 780), (1360, 850), width=4)
    draw.text((86, 1130), "VLM output is the single semantic source for SS conditioning and segmentation prompts.", fill=(85, 91, 101), font=F18)
    arrow(draw, (1585, 895), (1840, 895), width=4)

    # Right PartSeg main path.
    card(draw, (1885, 235, 2045, 430), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "voxel_whole.png", (1900, 260, 2030, 385))
    draw.text((1910, 394), "64³ voxel", fill=INK, font=F16)

    trapezoid(draw, (2080, 260, 2225, 410), "VAE\nEncoder\n❄", direction="narrow_right", fill=SAGE, outline=(96, 130, 99), fnt=F16)
    card(draw, (2260, 235, 2425, 430), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "latent_grid.png", (2278, 255, 2408, 388))
    draw.text((2288, 394), "latent grid 16³", fill=INK, font=F16)
    draw_seg_trunk(draw, (2470, 220, 2635, 455))
    trapezoid(draw, (2675, 260, 2820, 410), "VAE\nDecoder\n❄", direction="wide_right", fill=SAGE, outline=(96, 130, 99), fnt=F16)

    card(draw, (2855, 235, 3005, 430), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "voxel_parts.png", (2868, 255, 2992, 385))
    draw.text((2875, 394), "part voxels", fill=INK, font=F16)
    card(draw, (3030, 235, 3150, 430), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "parts_mesh.png", (3040, 252, 3140, 390))
    draw.text((3044, 394), "part meshes", fill=INK, font=F16)

    arrow(draw, (2045, 332), (2080, 332), width=4)
    arrow(draw, (2225, 332), (2260, 332), width=4)
    arrow(draw, (2425, 332), (2470, 332), width=4)
    arrow(draw, (2635, 332), (2675, 332), width=4)
    arrow(draw, (2820, 332), (2855, 332), width=4)
    arrow(draw, (3005, 332), (3030, 332), width=4)

    # Prompt path entering PartSeg.
    card(draw, (2140, 760, 2340, 875), "Prompt\nEncoder", fill=PINK, outline=(177, 102, 123), radius=12, fnt=F18)
    card(draw, (1920, 940, 2120, 1018), "Negative Prompt", fill=(255, 255, 255), outline=RED, radius=10, fnt=F16)
    new_badge(draw, 2070, 928)
    arrow(draw, (1840, 895), (2140, 820), width=4)
    arrow(draw, (2120, 978), (2240, 875), width=4)
    elbow(draw, [(2340, 818), (2515, 818), (2515, 455)], width=4)
    draw.text((1895, 848), "cross-panel part prompts", fill=LINE, font=F16)

    # Training-only supervision.
    dashed_round(draw, (2200, 930, 2925, 1190), 20, fill=(255, 255, 255), outline=(132, 140, 152), width=3)
    draw.text((2490, 947), "training only", fill=INK, font=F18)
    card(draw, (2235, 990, 2480, 1150), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "boundary_band.png", (2250, 1004, 2465, 1118))
    new_badge(draw, 2420, 982)
    draw.text((2278, 1122), "Boundary-band supervision", fill=INK, font=F16)
    card(draw, (2550, 990, 2890, 1150), None, fill=(255, 255, 255), outline=(164, 174, 184), radius=12)
    paste_img(canvas, ASSET / "corruption_pair.png", (2565, 1005, 2875, 1116))
    new_badge(draw, 2830, 982)
    draw.text((2608, 1122), "Structured voxel corruption", fill=INK, font=F16)
    arrow(draw, (2552, 455), (2360, 930), width=3, dashed=True)
    arrow(draw, (2552, 455), (2720, 930), width=3, dashed=True)

    # Residual and argmax.
    card(draw, (2875, 585, 3035, 655), "Body = residual", fill=(250, 250, 250), outline=(154, 163, 175), radius=10, fnt=F16)
    card(draw, (3048, 585, 3152, 655), "Argmax\n+ CC", fill=(250, 250, 250), outline=(154, 163, 175), radius=10, fnt=F16)
    elbow(draw, [(3090, 430), (3090, 515), (2955, 585)], width=4)
    elbow(draw, [(3090, 430), (3090, 585)], width=4)
    draw.text((3000, 674), "single owner", fill=LINE, font=F14)

    # Legend and downstream note.
    draw.text((1875, 1197), "per-part decode → sim-ready assets", fill=(88, 94, 104), font=F18)
    lx, ly = 2370, 1198
    draw.text((lx, ly), "❄ frozen", fill=BLUE, font=F16)
    draw.text((lx + 120, ly), "🔥 trainable", fill=AMBER, font=F16)
    new_badge(draw, lx + 270, ly - 4)
    draw.text((lx + 338, ly), "added vs baseline", fill=INK, font=F16)
    draw.rounded_rectangle((lx + 530, ly - 3, lx + 575, ly + 25), radius=7, outline=SOFT_LINE, width=2)
    draw.text((lx + 585, ly), "training only", fill=INK, font=F16)

    out = OUT / "pipeline_refstyle.png"
    canvas.save(out)
    canvas.save(OUT / "pipeline_refstyle_v2.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
