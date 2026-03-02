import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
#   Какой файл использовать (выберите один)
# ────────────────────────────────────────────────
# Вариант 1 — самый чистый и ограниченный топ-500
# FILE = "danbooru_general_unique_shuffled_cleaned_rare_removed_top_500.csv"

# Вариант 2 — после удаления редких, но ещё все теги
FILE = "danbooru_general_unique_shuffled_cleaned_rare_removed.csv"

# Вариант 3 — исходный очищенный от выбросов по кол-ву тегов
# FILE = "danbooru_general_unique_shuffled_cleaned.csv"

# Вариант 4 — самый исходный (до любых фильтров)
# FILE = "danbooru_general_unique_shuffled.csv"   # или даже оригинал

print(f"Читаем файл: {FILE}")
df = pd.read_csv(FILE)
print(f"Строк в датасете: {len(df):,}\n")

# ────────────────────────────────────────────────
#  Собираем все теги и считаем частоты
# ────────────────────────────────────────────────
all_tags = []
for tags_str in df['tags']:
    if pd.isna(tags_str) or not str(tags_str).strip():
        continue
    tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
    all_tags.extend(tags)

tag_counts = Counter(all_tags)
print(f"Уникальных тегов: {len(tag_counts):,}")

total_sum = sum(tag_counts.values())
print(f"Общее количество тегов (сумма частот): {total_sum:,}\n")

# Сортируем по убыванию частоты
sorted_freq = [freq for tag, freq in tag_counts.most_common()]

# Кумулятивная сумма
cumsum = np.cumsum(sorted_freq)

# Coverage в долях → в проценты
coverage = cumsum / total_sum * 100

# ────────────────────────────────────────────────
#  Вывод ключевых точек
# ────────────────────────────────────────────────
print("     N     |  Coverage   |  прирост от +100 тегов")
print("────────────┼─────────────┼───────────────────────")

prev_cov = 0
for n in range(50, 3001, 50):
    if n > len(coverage):
        break
    cov = coverage[n-1]
    delta = cov - prev_cov
    print(f"{n:8d}   |   {cov:5.2f}%   |     {delta:+5.2f}%")
    prev_cov = cov

# Точки, где покрытие переваливает важные значения
for target in [80, 85, 88, 90, 92, 93, 94, 95, 96, 97, 98]:
    idx = np.searchsorted(coverage, target)
    if idx < len(coverage):
        print(f"\n≥ {target}% покрытия → нужно топ-{idx+1} тегов (coverage = {coverage[idx]:.2f}%)")

# ────────────────────────────────────────────────
#  График
# ────────────────────────────────────────────────
plt.figure(figsize=(11, 6))

x = np.arange(1, len(coverage) + 1)

plt.plot(x, coverage, label='Cumulative coverage', color='#1f77b4', linewidth=2)

# Зона насыщения
plt.axhspan(90, 100, facecolor='gold', alpha=0.08)
plt.axhline(90, color='orange', linestyle='--', alpha=0.6, label='90% — часто достаточно')
plt.axhline(95, color='green', linestyle='--', alpha=0.6, label='95% — хороший компромисс')

plt.title("Покрытие тегов (Coverage) в зависимости от топ-N", fontsize=14, pad=12)
plt.xlabel("Количество самых популярных тегов (N)", fontsize=12)
plt.ylabel("Покрытие, %", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Ограничим ось x разумным диапазоном (часто 0–3000 хватает)
plt.xlim(0, min(3000, len(coverage)))
plt.ylim(0, 102)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
#  Топ-20 для понимания, кто "несёт" покрытие
# ────────────────────────────────────────────────
print("\nТоп-20 тегов по частоте и их индивидуальный вклад:")
top20 = tag_counts.most_common(20)
for i, (tag, cnt) in enumerate(top20, 1):
    percent = cnt / total_sum * 100
    print(f"{i:2d}. {tag:24} {cnt:7d} тегов → {percent:5.2f}%")