#!/usr/bin/env python3
"""
Перевод текста на изображениях (webp, png, jpg) с японского на русский.

Пайплайн:
  1. EasyOCR находит области с текстом (только детекция)
  2. Близкие рамки объединяются в "облачка"
  3. manga-ocr распознаёт японский текст (хорошо работает с вертикальным текстом)
  4. deep-translator переводит ja -> ru
  5. OpenCV стирает оригинальный текст (inpaint)
  6. Pillow вписывает русский перевод в то же место

Установка:
  pip install manga-ocr easyocr deep-translator opencv-python pillow numpy

Использование:
  python translate_webp.py page.webp
  python translate_webp.py ./input_folder -o ./output_folder --font ./fonts/AnimeAce.ttf
"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

#(всё, что умеет открывать Pillow)
EXTS = {".webp", ".png", ".jpg", ".jpeg", ".jpe", ".jfif",
        ".bmp", ".tif", ".tiff", ".gif", ".avif"}
QUALITY_EXTS = {".webp", ".jpg", ".jpeg", ".jpe", ".jfif", ".avif"}
JP_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")  # кана + кандзи

FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


# детекция
def detect_boxes(reader, img_rgb):
    """Возвращает список рамок [x1, y1, x2, y2]."""
    horizontal, free = reader.detect(
        img_rgb,
        canvas_size=2560,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
    )
    boxes = []
    for x1, x2, y1, y2 in horizontal[0]:
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    for poly in free[0]:
        pts = np.array(poly)
        boxes.append([int(pts[:, 0].min()), int(pts[:, 1].min()),
                      int(pts[:, 0].max()), int(pts[:, 1].max())])
    return boxes


def merge_boxes(boxes, gap):
    """Объединяет рамки, которые пересекаются или лежат ближе, чем gap пикселей.
    Если gap < 0 — склеивание отключено (для таблиц и словарей)."""
    if gap < 0:
        return [list(b) for b in boxes]
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if (a[0] - gap <= b[2] and b[0] - gap <= a[2] and
                        a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                    a = [min(a[0], b[0]), min(a[1], b[1]),
                         max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


# стирание
def erase_text(img_bgr, box, pad=4):
    """Стирает текст внутри рамки (inpaint только по тёмным/светлым пикселям текста)."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    bg_bright = np.median(gray) > 127
    flag = cv2.THRESH_BINARY_INV if bg_bright else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    mask_full = np.zeros(img_bgr.shape[:2], np.uint8)
    mask_full[y1:y2, x1:x2] = mask
    result = cv2.inpaint(img_bgr, mask_full, 3, cv2.INPAINT_TELEA)
    return result, bg_bright


# рендер текста
def load_font(path, size):
    if path:
        return ImageFont.truetype(path, size)
    for cand in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    sys.exit("Не найден шрифт с кириллицей. Укажите его через --font путь/к/шрифту.ttf")


def wrap(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def fit_text(text, box_w, box_h, font_path, max_size=40, min_size=9):
    """Подбирает максимальный размер шрифта, при котором текст влезает в рамку."""
    for size in range(max_size, min_size - 1, -1):
        font = load_font(font_path, size)
        lines = wrap(text, font, box_w)
        line_h = int(size * 1.15)
        too_wide = any(font.getlength(l) > box_w for l in lines)
        if len(lines) * line_h <= box_h and not too_wide:
            return font, lines, line_h
    font = load_font(font_path, min_size)
    return font, wrap(text, font, box_w), int(min_size * 1.15)


def draw_text(pil_img, box, text, font_path, dark_text):
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # русский текст длиннее японского, поэтому немного расширяем область
    ex, ey = int(bw * 0.2), int(bh * 0.1)
    x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
    x2, y2 = min(W, x2 + ex), min(H, y2 + ey)
    bw, bh = x2 - x1, y2 - y1

    font, lines, line_h = fit_text(text, bw, bh, font_path)
    draw = ImageDraw.Draw(pil_img)
    fill = (0, 0, 0) if dark_text else (255, 255, 255)
    outline = (255, 255, 255) if dark_text else (0, 0, 0)

    y = y1 + (bh - len(lines) * line_h) // 2
    for line in lines:
        x = x1 + (bw - font.getlength(line)) // 2
        draw.text((x, y), line, font=font, fill=fill,
                  stroke_width=1, stroke_fill=outline)
        y += line_h


# перевод
class GoogleTranslatorBackend:
    """Google Translate с паузой между запросами (легко ловит лимиты)."""

    def __init__(self, delay):
        from deep_translator import GoogleTranslator
        self.t = GoogleTranslator(source="ja", target="ru")
        self.delay = delay

    def translate(self, text):
        time.sleep(self.delay)
        return self.t.translate(text)


class NLLBBackend:
    """Локальная модель Meta NLLB: без лимитов и интернета (~2.5 ГБ при первом запуске)."""

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "facebook/nllb-200-distilled-600M"
        self.tok = AutoTokenizer.from_pretrained(name, src_lang="jpn_Jpan")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self.bos = self.tok.convert_tokens_to_ids("rus_Cyrl")

    def translate(self, text):
        inputs = self.tok(text, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=self.bos,
            max_length=200,
            num_beams=4,
        )
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


class ClaudeBackend:
    """Claude API: самый естественный перевод. Нужен ключ в ANTHROPIC_API_KEY."""

    def __init__(self, model):
        import os
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            sys.exit("Не задан ANTHROPIC_API_KEY. Установите переменную окружения.")
        self.client = anthropic.Anthropic()
        self.model = model

    def translate(self, text):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=("Ты переводчик манги. Переведи японскую реплику на русский: "
                    "естественно и коротко, как в речевом облачке. "
                    "Ответь только переводом, без пояснений."),
            messages=[{"role": "user", "content": text}],
        )
        return msg.content[0].text.strip()


def make_translator(kind, delay, claude_model="claude-sonnet-4-5"):
    if kind == "nllb":
        return NLLBBackend()
    if kind == "claude":
        return ClaudeBackend(claude_model)
    return GoogleTranslatorBackend(delay)


def translate(translator, text, retries=3):
    for i in range(retries):
        try:
            return translator.translate(text)
        except Exception as e:  # сеть/лимиты
            if i == retries - 1:
                print(f"  ! ошибка перевода: {str(e)[:200]}")
                return None
            time.sleep(3 * (i + 1))


# основной цикл
def process_image(path, out_path, reader, mocr, translator, font_path, gap, fmt="same"):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    boxes = merge_boxes(detect_boxes(reader, rgb), gap)
    print(f"  найдено областей: {len(boxes)}")

    items = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if (x2 - x1) < 12 or (y2 - y1) < 12:
            continue
        crop = pil.crop((max(0, x1 - 3), max(0, y1 - 3), x2 + 3, y2 + 3))
        jp = mocr(crop).strip()
        if not jp or not JP_RE.search(jp):
            continue
        ru = translate(translator, jp)
        if not ru:
            continue
        print(f"  {jp}  ->  {ru}")
        items.append((box, ru))

    # 1) стираем весь оригинальный текст
    bright_map = []
    for box, _ in items:
        bgr, bright = erase_text(bgr, box)
        bright_map.append(bright)

    # 2) вписываем перевод
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    for (box, ru), bright in zip(items, bright_map):
        draw_text(out, box, ru, font_path, dark_text=bright)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "png":
        out_path = out_path.with_suffix(".png")
        out.save(out_path)
    elif fmt == "jpg":
        out_path = out_path.with_suffix(".jpg")
        out.save(out_path, quality=95)
    elif fmt == "webp":
        out_path = out_path.with_suffix(".webp")
        out.save(out_path, quality=95)
    else:  # "same"
        save_kwargs = {"quality": 95} if out_path.suffix.lower() in QUALITY_EXTS else {}
        out.save(out_path, **save_kwargs)

    print(f"  сохранено: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Перевод текста на изображениях ja -> ru")
    ap.add_argument("input", help="файл изображения или папка")
    ap.add_argument("-o", "--output", default="translated", help="папка для результата")
    ap.add_argument("--font", default=None, help="путь к .ttf шрифту с кириллицей")
    ap.add_argument("--gap", type=int, default=25,
                    help="расстояние (px) для объединения рамок в одно облачко")
    ap.add_argument("--no-merge", action="store_true",
                    help="не склеивать соседние рамки (для таблиц, словарей)")
    ap.add_argument("--gpu", action="store_true", help="использовать GPU для EasyOCR")
    ap.add_argument("--translator", choices=["nllb", "google", "claude"], default="nllb",
                    help="nllb - локальная модель без лимитов (по умолчанию); "
                         "google - онлайн, с лимитами; claude - через API-ключ")
    ap.add_argument("--claude-model", default="claude-sonnet-4-5",
                    help="имя модели Claude (используется только с --translator claude)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="пауза между запросами к Google, секунд")
    ap.add_argument("--format", choices=["same", "png", "jpg", "webp"], default="same",
                    help="формат результата: same - как у исходника (gif сохраняется как png)")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    if not files:
        sys.exit("Изображения не найдены")

    import easyocr
    from manga_ocr import MangaOcr

    print("Загрузка моделей (первый запуск скачивает веса)...")
    reader = easyocr.Reader(["ja"], gpu=args.gpu)
    mocr = MangaOcr()
    translator = make_translator(args.translator, args.delay, args.claude_model)

    out_dir = Path(args.output)
    for f in files:
        print(f"\n{f.name}")
        gap = -1 if args.no_merge else args.gap
        process_image(f, out_dir / f.name, reader, mocr, translator,
                      args.font, gap, args.format)


if __name__ == "__main__":
    main()