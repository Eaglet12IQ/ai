import sqlite3
import csv
import itertools

# Загрузка слов из файлов skip_tags.txt и remove_tags.txt
def load_tags(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

skip_tags = load_tags("danbooru_api/skip_tags.txt")
remove_tags = load_tags("danbooru_api/remove_tags.txt")

# Подключение к базе данных SQLite
conn = sqlite3.connect("danbooru_api/danbooru_posts.db")
cursor = conn.cursor()

# Запрос для извлечения промптов
cursor.execute("SELECT tags FROM posts")
prompts = [row[0] for row in cursor.fetchall()]
conn.close()

# Функция для экранирования всех обратных слэшей
def escape_backslashes(tag):
    return tag.replace('\\', '\\\\')

# Функция для создания датасета
def create_dataset(prompts):
    dataset = []
    for prompt in prompts:
        # Разбиваем промпт на слова, убирая пробелы
        words = [word.strip().lower() for word in prompt.split(",")]
        
        # Проверяем, есть ли в промпте слова из skip_tags
        if any(word in skip_tags for word in words):
            continue  # Пропускаем весь промпт
        
        # Удаляем слова из remove_tags
        filtered_words = [word for word in words if word not in remove_tags]
        
        # Если после фильтрации промпт пустой, пропускаем его
        if not filtered_words:
            continue
        
        # Экранируем все обратные слэши
        filtered_words = [escape_backslashes(word) for word in filtered_words]
        filtered_prompt = ", ".join(filtered_words)
        
        # Создаем все комбинации тегов от 2 до len(filtered_words)
        for r in range(1, min(2, len(filtered_words)) + 1):
            for combo in itertools.combinations(filtered_words, r):
                dataset.append({
                    "input": ", ".join(combo),
                    "output": filtered_prompt
                })
    return dataset

# Создаем датасет
dataset = create_dataset(prompts)

# Сохраняем датасет в CSV
output_file = "prompt_dataset.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input", "output"])
    writer.writeheader()
    writer.writerows(dataset)

print(f"Датасет сохранен в {output_file}")
print(f"Всего записей: {len(dataset)}")

# Вывод первых нескольких записей для проверки
for entry in dataset[:5]:
    print(f"Вход: {entry['input']}, Выход: {entry['output']}")