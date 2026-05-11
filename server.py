import json
import os
import re
import ssl
import time
import threading
import urllib.parse
import urllib.request
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
import base64
from io import BytesIO

from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

PORT = 3000
HOST = "0.0.0.0"

# ------------------------------------------------------------
#  Модель тональности
# ------------------------------------------------------------
MODEL_NAME = os.environ.get("SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
MODEL_DIR = Path("./local_model").resolve()

print("Загрузка NLP-модели...")
try:
    if MODEL_DIR.exists():
        print(f"Загрузка модели из локальной папки: {MODEL_DIR}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        print(f"Локальная папка {MODEL_DIR} не найдена, загрузка модели: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    print("Модель загружена.")
except Exception as e:
    print(f"Не удалось загрузить NLP-модель: {e}")
    sentiment_pipeline = None

# ------------------------------------------------------------
# Словари для fallback
# ------------------------------------------------------------
POS_WORDS = [
    "good", "great", "excellent", "amazing", "love", "loved", "beautiful", "brilliant", "masterpiece",
    "pleasant", "wonderful", "engaging", "powerful", "best", "favorite", "fantastic", "insightful",
    "хорошо", "отлично", "прекрасно", "супер", "понравилось", "шедевр", "впечатляет", "сильный",
    "увлекательно", "замечательно", "классно", "душевно", "интересно", "потрясающе", "трогательно",
    "ярко", "захватывающе", "чудесно"
]

NEG_WORDS = [
    "bad", "terrible", "boring", "worst", "hate", "awful", "weak", "messy", "slow", "disappointing",
    "poor", "flat", "predictable", "confusing", "dull", "annoying", "ridiculous", "overrated",
    "плохо", "ужасно", "скучно", "отвратительно", "разочарование", "слита", "затянуто", "слабый",
    "слабая", "слабое", "неинтересно", "скучный", "скучная", "банально", "плоский", "плоская",
    "раздражает", "смазано", "сумбурно", "примитивно", "нудно", "разочаровал", "разочаровала"
]

THEME_KEYWORDS = {
    "Plot / Story": ["plot", "story", "narrative", "сюжет", "история", "книга", "повествование", "линия", "развитие"],
    "Characters": ["character", "characters", "hero", "heroes", "personage", "персонаж", "персонажи", "герой", "герои", "образ"],
    "Writing / Style": ["writing", "style", "prose", "language", "написан", "язык", "стиль", "слог", "перевод", "редактур"],
    "Pacing": ["pacing", "fast", "slow", "темп", "динамик", "затянут", "затянута", "затянуто", "ритм", "drag"],
    "Atmosphere": ["atmosphere", "mood", "tone", "атмосфер", "настроен", "мрач", "уют", "мир", "эмоци"],
    "Ending": ["ending", "end", "finale", "концов", "финал", "развязк", "эпилог"],
    "Worldbuilding": ["worldbuilding", "world", "setting", "world-buil", "мир", "сеттинг", "вселен", "лора"],
    "Emotion": ["emotion", "emotional", "touching", "heart", "чувств", "эмоц", "трог", "душев"],
    "Originality": ["original", "fresh", "unique", "originality", "оригин", "необыч", "свеж", "новиз", "клише"],
    "Romance / Relationships": ["romance", "love", "relationship", "роман", "любов", "отношен", "пара", "связь"],
    "Action / Suspense": ["action", "suspense", "thriller", "экшен", "напряж", "интриг", "триллер", "детектив", "саспенс"],
    "Humor": ["humor", "funny", "laugh", "юмор", "смешн", "ирони", "шутк"],
}

THEME_RU = {
    "Plot / Story": "Сюжет",
    "Characters": "Персонажи",
    "Writing / Style": "Стиль и язык",
    "Pacing": "Темп",
    "Atmosphere": "Атмосфера",
    "Ending": "Финал",
    "Worldbuilding": "Мир и сеттинг",
    "Emotion": "Эмоциональность",
    "Originality": "Оригинальность",
    "Romance / Relationships": "Романтика и отношения",
    "Action / Suspense": "Динамика и напряжение",
    "Humor": "Юмор",
}

def ru_theme(theme: str) -> str:
    return THEME_RU.get(theme, theme)

STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
    "как", "а", "то", "все", "она", "так", "его", "но", "да",
    "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут",
    "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего",
    "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти",
    "нас", "про", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед",
    "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
    "книга", "книги", "автор", "роман", "история", "очень",
    "просто", "это", "этого", "который", "которая", "которые",
    "book", "books", "author", "story", "novel", "read", "reading",
    "and", "or", "as", "the", "a", "an", "in", "on", "at", "to", "of", "for", "by", "with", "from", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "but", "if", "then",
    "than", "so", "do", "does", "did", "not", "no", "yes", "you", "your", "i", "me", "my", "we", "our",
    "they", "their", "he", "she", "them", "his", "her", "who", "whom", "which", "what", "when", "where", "why",
    "how", "also", "just", "very", "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "one", "two", "three", "new", "more", "most", "much", "many", "some", "any", "all", "each", "every", "other"
}

WORDCLOUD_STOPWORDS = set(WC_STOPWORDS) | STOP_WORDS

# ------------------------------------------------------------
# HTML-интерфейс 
# ------------------------------------------------------------
try:
    with open("index.html", "r", encoding="utf-8") as f:
        HTML_PAGE = f.read()
except FileNotFoundError:
    print("Файл index.html не найден. Убедитесь, что он лежит рядом с server.py")
    exit(1)

# ------------------------------------------------------------
#  HTTP-утилиты
# ------------------------------------------------------------
def get_html(url: str, delay: float = 0.8, prefer_encoding: str = None,
             extra_headers: dict = None) -> str:
    """
    Загружает страницу. extra_headers — дополнительные заголовки (Referer и т.п.).
    """
    time.sleep(delay)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
            raw = r.read()
            order = [prefer_encoding, "utf-8", "cp1251", "windows-1251"]
            for enc in order:
                if not enc:
                    continue
                try:
                    return raw.decode(enc, errors="ignore")
                except (UnicodeDecodeError, LookupError):
                    continue
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[get_html] Ошибка: {url} → {e}")
        return ""


def try_urls(urls: list, delay: float = 0.5, extra_headers: dict = None) -> tuple:
    """Пробует список URL по очереди, возвращает (html, url) первого успешного."""
    for url in urls:
        html = get_html(url, delay=delay, extra_headers=extra_headers)
        if html:
            return html, url
    return "", ""


def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_snippets(texts, limit=10, max_len=260):
    snippets = []
    for text in texts:
        clean = strip_html(text)
        if len(clean) >= 35:
            snippets.append(clean[:max_len])
        if len(snippets) >= limit:
            break
    return snippets

# ------------------------------------------------------------
#  Тональность и облако слов
# ------------------------------------------------------------
def _normalize_sentiment_label(label: str, score: float = None) -> str:
    """
    Преобразует разные форматы label к positive/neutral/negative.
    Поддерживает star-labels и классические positive/negative.
    """
    if not label:
        return "neutral"

    lbl = label.lower().strip()

    if "positive" in lbl:
        return "positive"
    if "negative" in lbl:
        return "negative"
    if "neutral" in lbl:
        return "neutral"

    m = re.search(r"(\d+)", lbl)
    if m:
        stars = int(m.group(1))
        if stars >= 4:
            return "positive"
        if stars == 3:
            return "neutral"
        return "negative"

    if score is not None:
        if score >= 0.65:
            return "positive"
        if score <= 0.35:
            return "negative"
        return "neutral"

    return "neutral"


def classify_review(text: str) -> str:
    low = (text or "").lower()

    if sentiment_pipeline is not None:
        try:
            res = sentiment_pipeline(low[:512], truncation=True)
            if isinstance(res, list) and res:
                item = res[0]
            elif isinstance(res, dict):
                item = res
            else:
                item = {}

            label = item.get("label", "")
            score = item.get("score")
            return _normalize_sentiment_label(label, score)
        except Exception:
            pass

    pos = sum(low.count(w) for w in POS_WORDS)
    neg = sum(low.count(w) for w in NEG_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def score_sentiment_counts(review_texts):
    """
    Батчевая классификация для ускорения и стабильности.
    """
    if not review_texts:
        return {"positive": 34, "neutral": 33, "negative": 33}, Counter()

    labels = ["neutral"] * len(review_texts)

    if sentiment_pipeline is not None:
        try:
            batch = [t[:512] for t in review_texts]
            results = sentiment_pipeline(batch, truncation=True, batch_size=16)
            for i, res in enumerate(results):
                label = res.get("label", "")
                score = res.get("score")
                labels[i] = _normalize_sentiment_label(label, score)
        except Exception:
            for i, text in enumerate(review_texts):
                labels[i] = classify_review(text)
    else:
        for i, text in enumerate(review_texts):
            labels[i] = classify_review(text)

    counts = Counter(labels)
    total = len(review_texts)
    positive = round((counts["positive"] / total) * 100)
    neutral = round((counts["neutral"] / total) * 100)
    negative = 100 - positive - neutral
    if negative < 0:
        negative = 0
        neutral = 100 - positive

    return {"positive": positive, "neutral": neutral, "negative": negative}, counts


def generate_wordcloud(review_texts, sentiment_type="positive"):
    texts = []

    for text in review_texts:
        polarity = classify_review(text)
        if polarity != sentiment_type:
            continue

        clean = strip_html(text).lower()
        clean = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        tokens = []
        for token in clean.split():
            token = token.strip().lower()
            if len(token) < 3:
                continue
            if token in WORDCLOUD_STOPWORDS:
                continue
            tokens.append(token)

        if tokens:
            texts.append(" ".join(tokens))

    if not texts:
        return None

    merged = " ".join(texts)

    try:
        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            stopwords=WORDCLOUD_STOPWORDS,
            collocations=False,
            max_words=120,
            min_font_size=10
        ).generate(merged)

        buffer = BytesIO()
        image = wc.to_image()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + encoded

    except Exception as e:
        print(f"[wordcloud] Ошибка: {e}")
        return None


def theme_counts(review_texts):
    grouped = {"positive": Counter(), "negative": Counter()}
    for text in review_texts:
        polarity = classify_review(text)
        if polarity not in ("positive", "negative"):
            continue
        low = text.lower()
        for theme, kws in THEME_KEYWORDS.items():
            hits = sum(low.count(kw) for kw in kws)
            if hits:
                grouped[polarity][theme] += hits
    return grouped


def _build_theme_list(counter_obj, max_items=5):
    total = sum(counter_obj.values()) or 1
    result = []
    for theme, count in counter_obj.most_common(max_items):
        score = max(20, int((count / total) * 100))
        result.append({"theme": ru_theme(theme), "score": score})
    return result


def _theme_phrase(theme: str, positive: bool):
    mapping = {
        "Plot / Story": ("Сюжет", "Сюжет"),
        "Characters": ("Персонажи", "Персонажи"),
        "Writing / Style": ("Стиль и язык", "Стиль и язык"),
        "Pacing": ("Темп", "Темп"),
        "Atmosphere": ("Атмосфера", "Атмосфера"),
        "Ending": ("Финал", "Финал"),
        "Worldbuilding": ("Мир и сеттинг", "Мир и сеттинг"),
        "Emotion": ("Эмоциональность", "Эмоциональность"),
        "Originality": ("Оригинальность", "Оригинальность"),
        "Romance / Relationships": ("Романтическая линия", "Романтическая линия"),
        "Action / Suspense": ("Динамика и напряжение", "Динамика и напряжение"),
        "Humor": ("Юмор", "Юмор"),
    }
    return mapping.get(theme, (theme, theme))[0 if positive else 1]


def _summary_from_themes(pos_list, neg_list, sentiment):
    pos_names = [x["theme"] for x in pos_list[:2]]
    neg_names = [x["theme"] for x in neg_list[:2]]
    if sentiment["positive"] >= sentiment["negative"]:
        lead = "Отзывы в целом склоняются к положительным"
    else:
        lead = "Отзывы заметно более критичные, чем похвальные"
    parts = [lead]
    if pos_names:
        parts.append("лучше всего отмечают " + ", ".join(_theme_phrase(t, True).lower() for t in pos_names))
    if neg_names:
        parts.append("чаще критикуют " + ", ".join(_theme_phrase(t, False).lower() for t in neg_names))
    return "; ".join(parts) + "."

# ------------------------------------------------------------
#  Анализ
# ------------------------------------------------------------
def local_analysis(reviews_data):
    all_reviews = []
    for item in reviews_data:
        for r in item.get("reviews", []):
            cleaned = strip_html(r)
            if cleaned:
                all_reviews.append(cleaned)

    if not all_reviews:
        return {
            "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
            "positive_themes": [],
            "negative_themes": [],
            "pros": [],
            "cons": [],
            "summary": "Недостаточно текстовых данных для уверенного анализа.",
            "review_counts": {"positive": 0, "neutral": 0, "negative": 0},
        }

    sentiment, counts = score_sentiment_counts(all_reviews)
    grouped = theme_counts(all_reviews)
    positive_themes = _build_theme_list(grouped["positive"])
    negative_themes = _build_theme_list(grouped["negative"])

    pros = [_theme_phrase(item["theme"], True) for item in positive_themes[:5]]
    cons = [_theme_phrase(item["theme"], False) for item in negative_themes[:5]]

    if not pros:
        pros = ["Позитивные оценки встречаются редко или выражены слабо"]
    if not cons:
        cons = ["Негативные оценки встречаются редко или выражены слабо"]

    summary = _summary_from_themes(positive_themes, negative_themes, sentiment)

    return {
        "sentiment": sentiment,
        "positive_themes": positive_themes,
        "negative_themes": negative_themes,
        "pros": pros[:5],
        "cons": cons[:5],
        "summary": summary,
        "review_counts": {
            "positive": counts["positive"],
            "neutral": counts["neutral"],
            "negative": counts["negative"],
        }
    }


def analyze_with_ai(reviews_data):
    return local_analysis(reviews_data)

# ------------------------------------------------------------
#  Парсинг
# ------------------------------------------------------------
def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_review_text(text: str) -> str:
    return strip_html(text or "")


def extract_reviews_from_html(html: str, selectors, min_len: int = 50, max_items: int = None):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for sel in selectors:
        blocks = soup.select(sel)
        if not blocks:
            continue
        for block in blocks:
            text = normalize_review_text(block.get_text(" ", strip=True))
            if len(text) >= min_len:
                results.append(text)
        if results:
            break

    if not results:
        for tag in soup.find_all(["article", "section", "blockquote", "div", "p"]):
            class_id = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
            if not any(k in class_id.lower() for k in ("review", "opinion", "response", "comment", "text")):
                continue
            text = normalize_review_text(tag.get_text(" ", strip=True))
            if len(text) >= min_len:
                results.append(text)

    results = dedupe_preserve_order(results)
    if max_items is not None:
        results = results[:max_items]
    return results


def collect_reviews_from_urls(urls, selectors, min_len: int = 50, max_items: int = 120,
                              delay: float = 0.5, extra_headers: dict = None):
    collected = []
    seen = set()
    for url in urls:
        html = get_html(url, delay=delay, extra_headers=extra_headers)
        if not html:
            continue
        page_reviews = extract_reviews_from_html(html, selectors, min_len=min_len)
        new_found = False
        for review in page_reviews:
            review = normalize_review_text(review)
            if len(review) < min_len or review in seen:
                continue
            seen.add(review)
            collected.append(review)
            new_found = True
            if len(collected) >= max_items:
                return collected
        if page_reviews and not new_found and collected:
            break
    return collected


def first_match_url(html: str, patterns, base: str = ""):
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            href = match.group(1)
            if href.startswith("http"):
                return href
            if base:
                return urllib.parse.urljoin(base, href)
            return href
    return ""


def extract_json_review_texts(data, limit: int = 120):
    texts = []
    seen = set()
    stack = [data]
    keys = {"text", "review", "content", "body", "message", "comment", "description"}

    while stack and len(texts) < limit:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, str) and k.lower() in keys:
                    cleaned = normalize_review_text(v)
                    if len(cleaned) >= 50 and cleaned not in seen:
                        seen.add(cleaned)
                        texts.append(cleaned)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
        elif isinstance(node, str):
            cleaned = normalize_review_text(node)
            if len(cleaned) >= 50 and cleaned not in seen:
                seen.add(cleaned)
                texts.append(cleaned)
    return texts

# ============================================================
#  ПАРСЕРЫ КНИЖНЫХ САЙТОВ
# ============================================================

def scrape_goodreads(query):
    q_enc = urllib.parse.quote(query)
    search_urls = [
        f"https://www.goodreads.com/search?q={q_enc}",
        f"https://www.goodreads.com/search?utf8=%E2%9C%93&q={q_enc}",
    ]
    search_html, _ = try_urls(search_urls, delay=0.2)
    if not search_html:
        return None

    candidate_links = re.findall(r'href="(/book/show/[^"#?]+(?:[^"]*)?)"', search_html)
    candidate_links = dedupe_preserve_order(candidate_links)
    if not candidate_links:
        return None

    book_url = urllib.parse.urljoin("https://www.goodreads.com", candidate_links[0])
    selectors = [
        "div.TruncatedContent__text",
        "section.ReviewText",
        "div.reviewText",
        "span.ReviewText",
        "article",
        "div[data-testid='review']",
        "div[class*='ReviewText']",
        "div[class*='reviewText']",
    ]

    reviews = []
    seen = set()
    for page in range(1, 9):
        page_url = book_url if page == 1 else f"{book_url}?page={page}"
        html = get_html(page_url, delay=0.4)
        if not html:
            if page == 1:
                return None
            break
        page_reviews = extract_reviews_from_html(html, selectors, min_len=60)
        new_added = False
        for review in page_reviews:
            if review not in seen:
                seen.add(review)
                reviews.append(review)
                new_added = True
        if not page_reviews:
            if page > 1:
                break
        elif not new_added and page > 1:
            break
        if len(reviews) >= 120:
            break

    return {"site": "Goodreads", "url": book_url, "reviews": reviews[:120]} if reviews else None


def scrape_fantlab(query):
    q_enc = urllib.parse.quote(query, encoding="utf-8")
    search_html, _ = try_urls([
        f"https://fantlab.ru/searchmain?searchstr={q_enc}&mode=works",
        f"https://fantlab.ru/search?q={q_enc}",
    ], delay=0.5)
    if not search_html:
        return None

    work_ids = dedupe_preserve_order(re.findall(r'href="/work(\d+)"', search_html))
    if not work_ids:
        return None

    work_id = work_ids[0]
    work_url = f"https://fantlab.ru/work{work_id}"
    selectors = [
        "div.response",
        "div[class*='response']",
        "div.review-text",
        "div.wform-text",
        "div[id^='response']",
        "article",
        "blockquote",
        "div[class*='review']",
    ]

    candidate_urls = [work_url, f"{work_url}/responses", f"{work_url}/otzyvy", f"{work_url}/reviews"]
    for page in range(2, 6):
        candidate_urls.extend([
            f"{work_url}?page={page}",
            f"{work_url}/responses?page={page}",
            f"{work_url}/responsespage{page}",
            f"{work_url}/otzyvy?page={page}",
            f"{work_url}/reviews?page={page}",
        ])

    reviews = collect_reviews_from_urls(candidate_urls, selectors, min_len=60, max_items=80, delay=0.5)
    return {"site": "Fantlab", "url": work_url, "reviews": reviews[:80]} if reviews else None


def scrape_labirint(query):
    q_enc = urllib.parse.quote(query, encoding="utf-8")
    search_urls = [
        f"https://www.labirint.ru/search/{q_enc}/?stype=0",
        f"https://www.labirint.ru/search/{q_enc}/",
    ]
    search_html, _ = try_urls(search_urls, delay=0.6)
    if not search_html:
        return None

    book_url = first_match_url(search_html, [r'href="(/books/\d+/)"'], base="https://www.labirint.ru")
    if not book_url:
        return None

    selectors = [
        "div.review",
        "div[class*='review']",
        "div.comment",
        "div[class*='comment']",
        "article",
        "blockquote",
        "p[itemprop='reviewBody']",
        "div[itemprop='reviewBody']",
    ]

    candidate_urls = [book_url]
    for page in range(2, 5):
        candidate_urls.append(f"{book_url}?page={page}")
        candidate_urls.append(f"{book_url}?display=reviews&page={page}")

    reviews = collect_reviews_from_urls(candidate_urls, selectors, min_len=60, max_items=60, delay=0.7)
    return {"site": "Labirint", "url": book_url, "reviews": reviews[:60]} if reviews else None

# ============================================================
#  Агрегация и анализ
# ============================================================
def analyze_item(item):
    reviews = item.get("reviews", [])
    analysis = analyze_with_ai([item])
    positive_cloud = generate_wordcloud(reviews, "positive")
    negative_cloud = generate_wordcloud(reviews, "negative")

    return {
        "site": item["site"],
        "url": item["url"],
        "review_count": len(reviews),
        "analysis": analysis,
        "wordclouds": {
            "positive": positive_cloud,
            "negative": negative_cloud
        }
    }


def search_all(query):
    thread_results = []
    lock = threading.Lock()

    def run(func):
        try:
            res = func(query)
            if res and res.get("reviews"):
                with lock:
                    thread_results.append(res)
        except Exception as e:
            print(f"[search_all] Ошибка в {func.__name__}: {e}")

    scrapers = [
        scrape_goodreads,
        scrape_fantlab,
        scrape_labirint,
    ]

    threads = [threading.Thread(target=run, args=(fn,), daemon=True) for fn in scrapers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=35)

    source_analyses = [analyze_item(item) for item in thread_results]
    overall_analysis = analyze_with_ai(thread_results)

    all_reviews = []
    for item in thread_results:
        all_reviews.extend(item.get("reviews", []))

    overall_positive_cloud = generate_wordcloud(all_reviews, "positive")
    overall_negative_cloud = generate_wordcloud(all_reviews, "negative")

    return {
        "query": query,
        "sources": source_analyses,
        "analysis": overall_analysis,
        "overall_analysis": overall_analysis,
        "overall_wordclouds": {
            "positive": overall_positive_cloud,
            "negative": overall_negative_cloud
        }
    }

# ============================================================
#  HTTP-сервер
# ============================================================
class LitHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._html(HTML_PAGE)
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/api/search":
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                q = data.get("query", "").strip()
                self._json(search_all(q))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "Not found"}, 404)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = HTTPServer((HOST, PORT), LitHandler)
    try:
        print(f"Сервер запущен на http://localhost:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()