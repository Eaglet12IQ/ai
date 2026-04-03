import pandas as pd
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt

def load_and_shuffle_dataset(file_path: str, random_state: int = 42) -> pd.DataFrame:
    """Загружает датасет и перемешивает строки"""
    print("Загружаем и перемешиваем датасет...")
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def shuffle_tags_in_dataframe(df: pd.DataFrame, tag_column: str = 'tags') -> pd.DataFrame:
    """Перемешивает теги внутри каждой строки"""
    def shuffle_tags(tags_str):
        if pd.isna(tags_str) or not str(tags_str).strip():
            return ""
        tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
        random.shuffle(tags)
        return ', '.join(tags)

    df = df.copy()
    df[tag_column] = df[tag_column].apply(shuffle_tags)
    return df


def remove_empty_tag_rows(df: pd.DataFrame, tag_column: str = 'tags') -> pd.DataFrame:
    """Удаляет строки, где после обработки тегов не осталось ни одного"""
    before = len(df)
    df = df[df[tag_column].str.strip() != ""].copy()
    removed = before - len(df)
    print(f"Удалено полностью пустых строк: {removed}")
    return df


def get_top_tags(df: pd.DataFrame, tag_column: str = 'tags', top_n: int = 1000) -> set:
    """Возвращает множество из топ-N самых частых тегов"""
    all_tags_flat = []
    for tags_str in df[tag_column]:
        tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
        all_tags_flat.extend(tags)

    tag_counts = Counter(all_tags_flat)
    print(f"\nУникальных тегов всего: {len(tag_counts):,}")

    top_tags_list = [tag for tag, _ in tag_counts.most_common(top_n)]
    print(f"Выбрано топ-{top_n} тегов")
    
    return set(top_tags_list)


def filter_tags_to_top(df: pd.DataFrame, top_tags: set, tag_column: str = 'tags') -> pd.DataFrame:
    """Оставляет только теги из топ-N"""
    def filter_to_top_tags(tags_str):
        if not str(tags_str).strip():
            return ""
        tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
        filtered = [t for t in tags if t in top_tags]
        return ', '.join(filtered) if filtered else ""

    df = df.copy()
    before = len(df)
    df[tag_column] = df[tag_column].apply(filter_to_top_tags)
    df = df[df[tag_column].str.strip() != ""].copy()

    removed = before - len(df)
    print(f"Строк после фильтрации по топ-тегам: {len(df):,} "
          f"(удалено {removed:,} опустевших строк)")
    
    return df


def deduplicate_by_tag_set(df: pd.DataFrame, tag_column: str = 'tags') -> pd.DataFrame:
    """Удаляет дубликаты по набору тегов (порядок не важен)"""
    print("\nУдаляем дубликаты по уникальному набору тегов...")

    def get_sorted_tags_set(tags_str):
        if not str(tags_str).strip():
            return frozenset()
        tags = [t.strip() for t in tags_str.split(', ') if t.strip()]
        return frozenset(tags)

    df = df.copy()
    df['tags_set'] = df[tag_column].apply(get_sorted_tags_set)

    before = len(df)
    df = df.drop_duplicates(subset=['tags_set'], keep='first').copy()
    after = len(df)

    removed = before - after
    percent = (removed / before * 100) if before > 0 else 0

    print(f"Удалено дубликатов: {removed:,} ({percent:.2f}%)")

    df = df.drop(columns=['tags_set'])
    return df


def filter_by_tag_count(df: pd.DataFrame, tag_column: str = 'tags',
                        low_quantile: float = 0.05, high_quantile: float = 0.95) -> pd.DataFrame:
    """Фильтрует строки по количеству тегов (убирает слишком короткие и слишком длинные)"""
    def count_tags(tags_str):
        if not str(tags_str).strip():
            return 0
        return len([t for t in str(tags_str).split(', ') if t.strip()])

    df = df.copy()
    df['tag_count'] = df[tag_column].apply(count_tags)

    print("\nСтатистика количества тегов:")
    print(df['tag_count'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    low_q = df['tag_count'].quantile(low_quantile)
    high_q = df['tag_count'].quantile(high_quantile)

    print(f"Применяем фильтр: {low_q:.1f} ≤ кол-во тегов ≤ {high_q:.1f}")

    before = len(df)
    df = df[(df['tag_count'] >= low_q) & (df['tag_count'] <= high_q)].copy()
    after = len(df)

    print(f"Удалено по длине тегов: {before - after:,} "
          f"({(before - after) / before * 100:.2f}%)")

    df = df.drop(columns=['tag_count'])
    return df


def save_cleaned_dataset(df: pd.DataFrame, top_n: int, output_dir: str = "."):
    """Сохраняет финальный датасет"""
    final_path = f"{output_dir}/danbooru_general_unique_shuffled_cleaned_top_{top_n}.csv"
    df = df[['tags']].copy()
    df.to_csv(final_path, index=False)

    print(f"\nФинальный датасет сохранён: {final_path}")
    print(f"Итоговый размер: {len(df):,} строк")
    return final_path


# ====================== Основная функция ======================
def main():
    random.seed(42)
    np.random.seed(42)  # на всякий случай

    FILE_PATH = 'danbooru_general_unique.csv'
    TOP_N = 1000

    df = load_and_shuffle_dataset(FILE_PATH)
    df = shuffle_tags_in_dataframe(df)
    df = remove_empty_tag_rows(df)

    top_tags = get_top_tags(df, top_n=TOP_N)
    df = filter_tags_to_top(df, top_tags)

    df = deduplicate_by_tag_set(df)
    df = filter_by_tag_count(df)

    save_cleaned_dataset(df, TOP_N)

if __name__ == "__main__":
    main()

# print("Строим график сравнения распределений длины...")

# plt.figure(figsize=(10, 6))

# plt.hist(tag_counts_before, bins=50, alpha=0.5, density=True, label="До удаления")
# plt.hist(tag_counts_after, bins=50, alpha=0.5, density=True, label="После удаления")

# plt.xlabel("Количество тегов в строке")
# plt.ylabel("Плотность")
# plt.title("Сравнение распределения длины промптов")
# plt.legend()

# plot_path = f"length_distribution_comparison_top_{TOP_N}.png"
# plt.savefig(plot_path, dpi=300)
# plt.close()

# print(f"✅ График сохранён: {plot_path}")

# # ─────────────────────────────────────────────────────────────
# # 2.6. Zipf-анализ распределения частот тегов
# # ─────────────────────────────────────────────────────────────
# print("\nСтроим Zipf-анализ распределения тегов...")

# from collections import Counter
# import numpy as np

# # Собираем финальные теги
# all_final_tags = []
# for tags_str in df['tags']:
#     tags = [t.strip() for t in tags_str.split(', ') if t.strip()]
#     all_final_tags.extend(tags)

# tag_freq = Counter(all_final_tags)

# print(f"Уникальных тегов после всех фильтров: {len(tag_freq):,}")
# print(f"Всего токенов: {len(all_final_tags):,}")

# # Сортировка по частоте
# sorted_tags = tag_freq.most_common()
# ranks = np.arange(1, len(sorted_tags) + 1)
# frequencies = np.array([freq for _, freq in sorted_tags])

# # ─────────────────────────────────────────
# # 1️⃣ Barplot топ-50
# # ─────────────────────────────────────────
# top_k = 1000
# top_tags = sorted_tags[:top_k]
# top_names = [tag for tag, _ in top_tags]
# top_values = [freq for _, freq in top_tags]

# plt.figure(figsize=(12, 6))
# plt.bar(range(top_k), top_values)
# plt.xticks(range(top_k), top_names, rotation=90)
# plt.title(f"Топ-{top_k} самых частых тегов")
# plt.ylabel("Частота")
# plt.tight_layout()

# barplot_path = f"top_{top_k}_tags_top_{TOP_N}.png"
# plt.savefig(barplot_path, dpi=300)
# plt.close()

# print(f"✅ Barplot сохранён: {barplot_path}")

# # ─────────────────────────────────────────
# # 2️⃣ Log-Log Zipf график
# # ─────────────────────────────────────────
# plt.figure(figsize=(8, 6))
# plt.loglog(ranks, frequencies)
# plt.xlabel("Rank")
# plt.ylabel("Frequency")
# plt.title("Zipf-проверка распределения тегов")
# plt.grid(True)

# zipf_path = f"zipf_distribution_top_{TOP_N}.png"
# plt.savefig(zipf_path, dpi=300)
# plt.close()

# print(f"✅ Zipf-график сохранён: {zipf_path}")

# # ─────────────────────────────────────────
# # 3️⃣ Оценка наклона (приближение закона Ципфа)
# # ─────────────────────────────────────────
# log_ranks = np.log(ranks)
# log_freq = np.log(frequencies)

# # Линейная регрессия в лог-пространстве
# slope, intercept = np.polyfit(log_ranks, log_freq, 1)

# print("\n=== Ключевые наблюдения по Zipf-распределению (для отчёта) ===")
# print(f"Тег №1 ('{top_tags[0][0]}') встречается в ~{top_tags[0][1]/len(df)*100:.1f}% промптов")
# print(f"Тег №10 — в ~{top_tags[9][1]/len(df)*100:.1f}% промптов")
# print(f"Тег №100 — примерно в {top_tags[99][1]/len(df)*100:.2f}% промптов")
# print(f"Тег №500 — примерно в {top_tags[499][1]/len(df)*100:.2f}% промптов")
# print(f"Тег №1000 — примерно в {top_tags[999][1]/len(df)*100:.3f}% промптов")
# print(f"Наклон log-log регрессии: {slope:.4f}  (близко к классическому Zipf −1)")
# print("Вывод: распределение сильно скошенное → 80–90% всех упоминаний тегов приходится на первые 200–400 элементов.")
# print("Это оправдывает выбор TOP_N ≈ 800–1200 как разумный компромисс.")

# # ─────────────────────────────────────────────────────────────
# # 2.7. Co-occurrence Matrix (Jaccard-нормализация)
# # ─────────────────────────────────────────────────────────────
# print("\nСтроим Co-occurrence matrix (Jaccard)...")

# import numpy as np

# COOC_TOP_K = 30

# top_cooc_tags = [tag for tag, _ in tag_freq.most_common(COOC_TOP_K)]
# tag_to_idx = {tag: i for i, tag in enumerate(top_cooc_tags)}

# # Матрица совместных появлений
# cooc_matrix = np.zeros((COOC_TOP_K, COOC_TOP_K), dtype=np.int32)

# # Отдельно считаем частоту каждого тега
# tag_counts_top = {tag: tag_freq[tag] for tag in top_cooc_tags}

# # Заполняем матрицу пересечений
# for tags_str in df['tags']:
#     tags = [t.strip() for t in tags_str.split(', ') if t.strip()]
#     filtered = list(set([t for t in tags if t in tag_to_idx]))

#     for i in range(len(filtered)):
#         for j in range(i, len(filtered)):
#             idx_i = tag_to_idx[filtered[i]]
#             idx_j = tag_to_idx[filtered[j]]
#             cooc_matrix[idx_i, idx_j] += 1
#             if i != j:
#                 cooc_matrix[idx_j, idx_i] += 1

# # ─────────────────────────────────────────
# # Jaccard нормализация
# # ─────────────────────────────────────────
# jaccard_matrix = np.zeros_like(cooc_matrix, dtype=float)

# for i in range(COOC_TOP_K):
#     for j in range(COOC_TOP_K):
#         if i == j:
#             jaccard_matrix[i, j] = 1.0
#         else:
#             intersection = cooc_matrix[i, j]
#             union = (
#                 tag_counts_top[top_cooc_tags[i]]
#                 + tag_counts_top[top_cooc_tags[j]]
#                 - intersection
#             )
#             if union > 0:
#                 jaccard_matrix[i, j] = intersection / union

# # ─────────────────────────────────────────
# # Убираем диагональ (чтобы не доминировала)
# # ─────────────────────────────────────────
# np.fill_diagonal(jaccard_matrix, 0)

# # ─────────────────────────────────────────
# # Построение heatmap
# # ─────────────────────────────────────────
# plt.figure(figsize=(12, 10))
# plt.imshow(jaccard_matrix)
# plt.colorbar()

# plt.xticks(range(COOC_TOP_K), top_cooc_tags, rotation=90)
# plt.yticks(range(COOC_TOP_K), top_cooc_tags)

# plt.title(f"Co-occurrence Matrix (Jaccard, Top-{COOC_TOP_K})")
# plt.tight_layout()

# cooc_path = f"cooccurrence_jaccard_top_{COOC_TOP_K}_of_{TOP_N}.png"
# plt.savefig(cooc_path, dpi=300)
# plt.close()

# print(f"✅ Co-occurrence (Jaccard) сохранена: {cooc_path}")

# print("\n=== Ключевые наблюдения по Jaccard-матрице co-occurrence (top-30) ===")
# print("Наиболее часто встречающиеся вместе пары (Jaccard > 0.40):")

# # Собираем топ-пары
# pairs = []
# for i in range(COOC_TOP_K):
#     for j in range(i+1, COOC_TOP_K):
#         jac = jaccard_matrix[i,j]
#         if jac > 0.35:  # порог можно поднять/опустить
#             pairs.append((jac, top_cooc_tags[i], top_cooc_tags[j]))

# pairs.sort(reverse=True)

# for jac, t1, t2 in pairs[:15]:  # топ-15 самых сильных связей
#     print(f"{jac:.3f}  —  {t1:20} ↔ {t2}")