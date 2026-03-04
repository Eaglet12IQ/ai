import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Фиксация воспроизводимости
# ─────────────────────────────────────────────────────────────
random.seed(42)

# ─────────────────────────────────────────────────────────────
# 0. Загрузка и перемешивание
# ─────────────────────────────────────────────────────────────
print("Загружаем и перемешиваем датасет...")
df = pd.read_csv('danbooru_general_unique.csv')

# Перемешиваем строки
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Перемешиваем теги внутри строк
def shuffle_tags(tags_str):
    if pd.isna(tags_str) or not str(tags_str).strip():
        return ""
    tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
    random.shuffle(tags)
    return ', '.join(tags)

df['tags'] = df['tags'].apply(shuffle_tags)

# Удаляем сразу полностью пустые строки (если появились)
before_empty = len(df)
df = df[df['tags'].str.strip() != ""].copy()
print(f"Удалено полностью пустых строк: {before_empty - len(df)}")

# ─────────────────────────────────────────────────────────────
# 1. Обрезка до топ-N (самый важный шаг)
# ─────────────────────────────────────────────────────────────
print("Считаем частоты тегов для выбора топ-N...")

all_tags_flat = []
for tags_str in df['tags']:
    tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
    all_tags_flat.extend(tags)

tag_counts = Counter(all_tags_flat)
print(f"Уникальных тегов всего: {len(tag_counts):,}")

TOP_N = 1000          # ← меняй здесь для разных экспериментов: 500, 800, 1000, 1500 и т.д.
top_tags_list = [tag for tag, _ in tag_counts.most_common(TOP_N)]
top_tags = set(top_tags_list)

print(f"\nБерём топ-{TOP_N} тегов")
print("Топ-10:", ", ".join(top_tags_list[:10]))

# Функция фильтрации тегов внутри строки
def filter_to_top_tags(tags_str):
    if not str(tags_str).strip():
        return ""
    tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
    filtered = [t for t in tags if t in top_tags]
    if not filtered:
        return ""
    return ', '.join(filtered)

print("\nПрименяем фильтрацию тегов до топ-N...")
before_filter = len(df)
df['tags'] = df['tags'].apply(filter_to_top_tags)

# Удаляем строки, которые опустели после фильтрации
df = df[df['tags'].str.strip() != ""].copy()
print(f"Строк после удаления опустевших: {len(df):,} (было {before_filter:,}, удалено {before_filter - len(df):,})\n")

# ─────────────────────────────────────────────────────────────
# 1.5. Удаление дубликатов по набору тегов (игнорируя порядок)
# ─────────────────────────────────────────────────────────────
print("\nУдаляем дубликаты по уникальному набору тегов (без учёта порядка)...")
# Создаём столбец с отсортированным набором тегов (как frozenset для уникальности)
def get_sorted_tags_set(tags_str):
    if not tags_str.strip():
        return frozenset()
    tags = [t.strip() for t in tags_str.split(', ') if t.strip()]
    return frozenset(tags)  # frozenset для хэшируемости и игнора порядка
df['tags_set'] = df['tags'].apply(get_sorted_tags_set)
before_dedup = len(df)
# Удаляем дубликаты по 'tags_set', сохраняя первое вхождение
df = df.drop_duplicates(subset=['tags_set'], keep='first').copy()
after_dedup = len(df)
print(f"Было строк: {before_dedup:,}")
print(f"Стало строк: {after_dedup:,}")
diff_dedup = before_dedup - after_dedup
percent_dedup = (diff_dedup / before_dedup * 100) if before_dedup > 0 else 0
print(f"Удалено дубликатов: {diff_dedup:,} ({percent_dedup:.2f}%)\n")
# Удаляем вспомогательный столбец
df = df.drop(columns=['tags_set'])

# ─────────────────────────────────────────────────────────────
# 2. Фильтрация по длине — ТОЛЬКО ТЕПЕРЬ
# ─────────────────────────────────────────────────────────────
print("Считаем длины последовательностей...")

def count_tags(tags_str):
    if not str(tags_str).strip():
        return 0
    return len([t for t in str(tags_str).split(', ') if t.strip()])

df['tag_count'] = df['tags'].apply(count_tags)

print("Статистика длин после обрезки и очистки:")
print(df['tag_count'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

# Сохраняем распределение ДО фильтра
tag_counts_before = df['tag_count'].copy()

low_q = df['tag_count'].quantile(0.05)
high_q = df['tag_count'].quantile(0.95)

print(f"\nПрименяем фильтр: {low_q:.1f} ≤ кол-во тегов ≤ {high_q:.1f}")

before_len = len(df)
df = df[(df['tag_count'] >= low_q) & (df['tag_count'] <= high_q)].copy()
after_len = len(df)

# Сохраняем распределение ПОСЛЕ фильтра
tag_counts_after = df['tag_count'].copy()

print(f"Было строк: {before_len:,}")
print(f"Стало строк: {after_len:,}")
print(f"Удалено по длине: {before_len - after_len:,} "
      f"({(before_len - after_len) / before_len * 100:.2f}%)\n")

# ─────────────────────────────────────────────────────────────
# 2.5. Сохранение overlay-графика
# ─────────────────────────────────────────────────────────────
print("Строим график сравнения распределений длины...")

plt.figure(figsize=(10, 6))

plt.hist(tag_counts_before, bins=50, alpha=0.5, density=True, label="До удаления")
plt.hist(tag_counts_after, bins=50, alpha=0.5, density=True, label="После удаления")

plt.xlabel("Количество тегов в строке")
plt.ylabel("Плотность")
plt.title("Сравнение распределения длины промптов")
plt.legend()

plot_path = f"length_distribution_comparison_top_{TOP_N}.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ График сохранён: {plot_path}")

# ─────────────────────────────────────────────────────────────
# 3. Сохранение результата
# ─────────────────────────────────────────────────────────────
final_path = f"danbooru_general_unique_shuffled_cleaned_top_{TOP_N}.csv"
df = df[['tags']]  # оставляем только нужную колонку
df.to_csv(final_path, index=False)

print(f"✅ Финальный датасет сохранён: {final_path}")
print(f"Итоговый размер: {len(df):,} строк")
print(f"Готов к использованию для топ-{TOP_N}")