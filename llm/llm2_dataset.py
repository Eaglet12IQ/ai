import sqlite3
import csv
from pathlib import Path

# ────────────────────────────────────────────────
#  Настройки — можно менять здесь
# ────────────────────────────────────────────────
SKIP_IF_LESS_THAN_N_TAGS = 4    # пропускаем изображения с очень малым кол-вом тегов

# пути (можно вынести в аргументы или конфиг позже)
DB_PATH = "danbooru_api/danbooru_posts.db"
SKIP_TAGS_PATH = "danbooru_api/skip_tags.txt"
REMOVE_TAGS_PATH = "danbooru_api/remove_tags.txt"
OUTPUT_CSV = "llm/unique_prompts.csv"


def load_tags(file_path: str) -> set[str]:
    path = Path(file_path)
    if not path.is_file():
        print(f"Warning: file not found → {path}")
        return set()
    
    normalized = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            norm = normalize_tag(line.lower())
            if norm:
                normalized.add(norm)
    return normalized


def normalize_tag(tag: str) -> str:
    """Приводим тег из txt к виду с экранированными скобками"""
    tag = tag.replace("_", " ")
    tag = tag.strip()
    return tag


def remove_backslashes(s: str) -> str:
    return s.replace("\\", "")


def create_unique_prompts() -> list[dict]:
    skip_tags   = load_tags(SKIP_TAGS_PATH)
    remove_tags = load_tags(REMOVE_TAGS_PATH)
    
    print(f"Loaded {len(skip_tags):,} skip tags")
    print(f"Loaded {len(remove_tags):,} remove tags")
    
    if not skip_tags and not remove_tags:
        print("Warning: both skip_tags and remove_tags are empty")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT tags FROM posts")
    raw_prompts = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Total raw records in DB: {len(raw_prompts):,}")

    unique_outputs = set()
    skipped_count = 0
    too_few_tags_count = 0

    for i, raw in enumerate(raw_prompts, 1):
        if i % 50000 == 0:
            print(f"Processed {i:,} / {len(raw_prompts):,} prompts")

        tags = [remove_backslashes(t.strip().lower()) for t in raw.split(",") if t.strip()]

        # Проверка skip
        if any(t in skip_tags for t in tags):
            skipped_count += 1
            continue

        # Фильтрация remove
        filtered = [t for t in tags if t not in remove_tags]

        # Сохраняем как строку — именно в том виде, в котором будет output
        full_prompt = ", ".join(filtered)
        
        # Добавляем в множество → автоматически убираются дубликаты
        unique_outputs.add(full_prompt)

    print(f"  Skipped by skip_tags       : {skipped_count:,}")
    print(f"  Skipped by too few tags    : {too_few_tags_count:,}")
    print(f"  Unique clean prompts       : {len(unique_outputs):,}")

    # Преобразуем в список словарей с одним полем
    dataset = [{"output": prompt} for prompt in unique_outputs]
    return dataset


def main():
    print("Loading and processing dataset...")
    dataset = create_unique_prompts()
    
    print(f"\nFinal dataset size: {len(dataset):,} unique prompts")
    
    if dataset:
        print("\nFirst 4 examples:")
        for ex in dataset[:4]:
            out = ex["output"]
            preview = out[:180] + "..." if len(out) > 180 else out
            print(f"  {preview}")
            print()

    # Сохранение
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["output"])
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f"Dataset saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()