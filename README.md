# Manga Translator JA → RU

Скрипт находит японский текст на изображениях (webp, png, jpg), стирает его
и вписывает русский перевод на то же место.

## Возможности
- EasyOCR + manga-ocr для распознавания (включая вертикальный текст)
- Три переводчика: NLLB (локально), Google, Claude API
- Поддержка webp, png, jpg, jpeg, bmp, tiff, gif
- Работа с отдельным файлом или папкой глав

## Требования
- Python 3.10–3.12 (на 3.13/3.14 часть библиотек может не работать)
- 8+ ГБ RAM
- ~2.5 ГБ диска для модели NLLB
- GPU опционально: флаг `--gpu`

## Установка
```bash
git clone https://github.com/Miksistt/manga-translator.git
cd manga-translator

python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/macOS

# базовые зависимости + NLLB (используется по умолчанию)
pip install -r requirements-nllb.txt

# или, если хотите только Google-перевод (легче, без torch):
# pip install -r requirements.txt

# или, если хотите Claude:
# pip install -r requirements-claude.txt
```

## Использование
```bash
# одна страница
python translate.py page.webp

# папка с главами -> папка translated
python translate.py ./chapters -o ./translated

# переводчики
python translate.py . --translator nllb       # по умолчанию
python translate.py . --translator claude
python translate.py . --translator google --delay 2
```

## Опции
| Опция | Значение |
|---|---|
| `input` | файл или папка |
| `-o, --output` | папка результата (по умолчанию `translated`) |
| `--font` | путь к .ttf с кириллицей |
| `--gap` | склейка облачек, px (по умолчанию 25) |
| `--gpu` | GPU для EasyOCR |
| `--translator` | `nllb` / `google` / `claude` |
| `--delay` | пауза между запросами Google, сек |
| `--format` | `same` / `png` / `jpg` / `webp` |
| `--claude-model` | имя модели Claude |

## Переводчики
- **nllb** — локально, без интернета и лимитов. Модель Meta `nllb-200-distilled-600M`
  распространяется под **CC-BY-NC-4.0** (некоммерческая).
- **google** — через `deep-translator`, неофициальный доступ. Легко ловит капчу.
  Используйте `--delay 2` и не запускайте большими пачками.
- **claude** — нужен ключ в переменной `ANTHROPIC_API_KEY`.
  ```bat
  set ANTHROPIC_API_KEY=sk-ant-...
  python translate.py . --translator claude
  ```

## Шрифты
Нужен `.ttf` с кириллицей. Пример:
```bat
python translate.py . --font C:\Windows\Fonts\comic.ttf
```
По умолчанию скрипт сам найдёт Arial/DejaVu.

## Ограничения
- Звуковые эффекты на фоне стираются неидеально.
- Google-перевод literal, для художественного текста нужна вычитка.
- GIF обрабатывается как один кадр.

## Лицензия
Код — MIT. Модели и зависимости имеют свои лицензии, см.
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Дисклеймер
Инструмент предназначен для личного использования с законно полученными
изображениями. Автор не поддерживает пиратство и распространение защищённых
авторским правом материалов.