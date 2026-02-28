import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import os
import re

# ─────────────────────────────────────────────────────────────
# 0. Создание копии датасета с перемешиванием строк + тегов внутри строки
# ─────────────────────────────────────────────────────────────
print("Создаём копию датасета с перемешиванием строчек и тегов внутри строк...")

df = pd.read_csv('danbooru_general_unique.csv')

# 1. Перемешиваем сами строки (примеры) между собой
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Перемешиваем теги внутри каждой строки
def shuffle_tags_in_row(tags_str):
    if pd.isna(tags_str) or not str(tags_str).strip():
        return tags_str
    tags = [t.strip() for t in str(tags_str).split(', ') if t.strip()]
    random.shuffle(tags)
    return ', '.join(tags)

df_shuffled['tags'] = df_shuffled['tags'].apply(shuffle_tags_in_row)

# Сохраняем копию
shuffled_path = 'danbooru_general_unique_shuffled.csv'
df_shuffled.to_csv(shuffled_path, index=False)
print(f"✓ Копия успешно создана и сохранена: {shuffled_path}")
print(f"   Размер: {len(df_shuffled):,} строк")

# Используем копию дальше (остальной код без изменений)
df = df_shuffled