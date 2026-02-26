import requests
import time
import csv
import random
from urllib.parse import quote
from pathlib import Path
from typing import Set, List

# ─── вспомогательные функции (без изменений) ────────────────────────────────

def load_tags_from_file(filename: str) -> Set[str]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return {tag.strip() for tag in f if tag.strip()}
    except FileNotFoundError:
        print(f"Файл {filename} не найден → пустой набор")
        return set()


def normalize_tags(tags_str: str) -> frozenset:
    if not tags_str:
        return frozenset()
    return frozenset(t.strip() for t in tags_str.split(',') if t.strip())


def format_tags(character_tags: str, general_tags: str, remove_tags: Set[str]) -> str:
    tags = ' '.join([character_tags, general_tags]).strip()
    if not tags:
        return ""
    tag_list = tags.split()
    filtered = [t for t in tag_list if t not in remove_tags]
    formatted = []
    for tag in filtered:
        tag = tag.replace('_', ' ')
        if '(' in tag and ')' in tag:
            tag = tag.replace('(', r'\(').replace(')', r'\)')
        formatted.append(tag)
    return ', '.join(formatted)


def fetch_danbooru_posts(tags_query: str, page: int, limit=100) -> list:
    url = f"https://danbooru.donmai.us/posts.json?tags={quote(tags_query)}&page={page}&limit={limit}"
    headers = {'User-Agent': 'DanbooruSequentialFetcher/0.5'}

    proxy_str = "http://aTrW6z5K6K:K3sRNZPdjI@193.124.133.137:21318"
    proxies = {"http": proxy_str, "https": proxy_str}

    auth = ('sunsiutaAI', 'm5vtFYic7ZH4vFM2jZ8gMGYs')

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, headers=headers, proxies=proxies, auth=auth, timeout=16)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            else:
                print("Не список в ответе API")
                return []
        except Exception as e:
            print(f"[попытка {attempt}/{max_attempts}] ошибка: {e}")
            if attempt < max_attempts:
                time.sleep(10 + random.uniform(0, 8))
    print("Страница не загрузилась после всех попыток")
    return []


def should_accept_post(char_tags_str: str, allowed_chars: Set[str]) -> bool:
    if not char_tags_str.strip():
        return True
    char_set = set(char_tags_str.split())
    return char_set.issubset(allowed_chars)


# ─── основная функция ───────────────────────────────────────────────────────

def sequential_per_character_then_general(
    csv_path: str = "danbooru_general_unique.csv",
    batch_save_every: int = 200,
):
    allowed_chars   = load_tags_from_file('danbooru_api/character_tags.txt')
    skip_tags       = load_tags_from_file('danbooru_api/skip_tags.txt')
    remove_tags     = load_tags_from_file('danbooru_api/remove_tags.txt')

    if not allowed_chars:
        print("Нет персонажей → выход")
        return

    print(f"Разрешённых персонажей: {len(allowed_chars):,d}")
    print(f"Skip / Remove тегов  : {len(skip_tags)} / {len(remove_tags)}")

    csv_file = Path(csv_path)
    fieldnames = ['tags']
    seen_tag_sets: Set[frozenset] = set()

    total_saved = 0
    if csv_file.exists():
        with csv_file.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm = normalize_tags(row.get('tags', ''))
                if norm:
                    seen_tag_sets.add(norm)
        total_saved = len(seen_tag_sets)
        print(f"Уже есть уникальных наборов: {total_saved:,d}")

    batch: List[dict] = []

    # ─── Фаза 1: по каждому персонажу до конца страниц ──────────────────────
    print("\n=== Фаза 1: парсинг по каждому персонажу до исчерпания страниц ===")

    for character in sorted(allowed_chars):  # сортировка для воспроизводимости
        print(f"\n→ Персонаж: {character}")
        page = 1
        character_exhausted = False

        while not character_exhausted:
            query = f"{character} rating:g"
            posts = fetch_danbooru_posts(query, page)

            if not posts:
                print(f"  → страниц больше нет (стр {page})")
                character_exhausted = True
                break

            new_in_page = 0

            for post in posts:
                char_tags = post.get('tag_string_character', '').strip()
                gen_tags  = post.get('tag_string_general', '').strip()

                if not should_accept_post(char_tags, allowed_chars):
                    continue

                all_tags_set = set(char_tags.split() + gen_tags.split())
                if skip_tags & all_tags_set:
                    continue

                formatted = format_tags(char_tags, gen_tags, remove_tags)
                if not formatted:
                    continue

                norm_set = normalize_tags(formatted)
                if norm_set in seen_tag_sets:
                    continue

                batch.append({'tags': formatted})
                seen_tag_sets.add(norm_set)
                new_in_page += 1
                total_saved += 1

            if new_in_page > 0:
                print(f"  стр {page:4d} → +{new_in_page:3d}  (всего: {total_saved:,d})")

            if batch and len(batch) >= batch_save_every:
                _save_batch(csv_file, batch, fieldnames)
                batch.clear()

            page += 1
            time.sleep(2.4 + random.uniform(0, 2.0))

    print("\nВсе персонажи пройдены до конца → переходим в общий режим")

    # ─── Фаза 2: бесконечный общий режим rating:g ───────────────────────────
    print("=== Фаза 2: бесконечный сбор rating:g (без привязки к персонажу) ===")

    page = 1
    while True:
        try:
            posts = fetch_danbooru_posts("rating:g", page)

            if not posts:
                print(f"Общая стр {page} пустая → пауза 120 сек")
                time.sleep(120)
                page += 1
                continue

            new_in_page = 0

            for post in posts:
                char_tags = post.get('tag_string_character', '').strip()
                gen_tags  = post.get('tag_string_general', '').strip()

                if not should_accept_post(char_tags, allowed_chars):
                    continue

                all_tags_set = set(char_tags.split() + gen_tags.split())
                if skip_tags & all_tags_set:
                    continue

                formatted = format_tags(char_tags, gen_tags, remove_tags)
                if not formatted:
                    continue

                norm_set = normalize_tags(formatted)
                if norm_set in seen_tag_sets:
                    continue

                batch.append({'tags': formatted})
                seen_tag_sets.add(norm_set)
                new_in_page += 1
                total_saved += 1

            if new_in_page > 0:
                print(f"[общий] стр {page:6d} → +{new_in_page:4d}  (всего: {total_saved:,d})")

            if batch and len(batch) >= batch_save_every:
                _save_batch(csv_file, batch, fieldnames)
                batch.clear()

            page += 1
            time.sleep(2.7 + random.uniform(0, 2.2))

        except KeyboardInterrupt:
            if batch:
                _save_batch(csv_file, batch, fieldnames)
                batch.clear()
            print(f"\nОстановлено. Всего уникальных: {total_saved:,d}")
            break
        except Exception as e:
            print(f"Ошибка в общем режиме стр {page}: {e}")
            time.sleep(150)


def _save_batch(csv_file: Path, batch: List[dict], fieldnames: List[str]):
    mode = 'a' if csv_file.exists() else 'w'
    with csv_file.open(mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        writer.writerows(batch)
    print(f"  → сохранено {len(batch)} строк")


if __name__ == '__main__':
    sequential_per_character_then_general(
        csv_path="danbooru_general_unique.csv",
        batch_save_every=250,
    )