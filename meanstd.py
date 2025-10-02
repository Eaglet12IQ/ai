import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# Определение BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_groups_from_txt(file_path):
    """Чтение списка групп из txt файла"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, "r") as f:
        groups = [line.strip() for line in f.readlines() if line.strip()]
    return groups

def compute_dataset_statistics(dataset_path, groups=None, img_size=224):
    """
    Вычисляет среднее и стандартное отклонение по каналам RGB, обрабатывая все изображения по одному.
    
    Args:
        dataset_path (str): Путь к датасету.
        groups (list, optional): Список групп. Если None, все группы.
        img_size (int): Размер для ресайза (по умолчанию 224).
    
    Returns:
        tuple: (mean, std) - средние значения и стандартные отклонения по каналам RGB.
    """
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sums_squares = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    image_count = 0
    low_diversity_images = 0

    if groups is None:
        groups = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for group in tqdm(groups, desc="Обработка групп"):
        group_path = os.path.join(dataset_path, group)
        if not os.path.isdir(group_path):
            continue
        
        png_files = [f for f in os.listdir(group_path) if f.endswith('.png')]
        
        for file in png_files:
            img_path = os.path.join(group_path, file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))  # Ресайз до 224x224
                img_array = np.array(img).astype(np.float64) / 255.0
                
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    print(f"Предупреждение: NaN или Inf в {img_path}")
                    continue
                
                # Проверка цветового разнообразия
                channel_diff = np.abs(img_array[:, :, 0] - img_array[:, :, 1]).mean() + \
                             np.abs(img_array[:, :, 1] - img_array[:, :, 2]).mean()
                if channel_diff < 1e-5:
                    print(f"Предупреждение: {img_path} имеет низкое цветовое разнообразие (разница каналов: {channel_diff})")
                    low_diversity_images += 1
                
                num_pixels = img_array.shape[0] * img_array.shape[1]
                channel_sums += np.sum(img_array, axis=(0, 1))
                channel_sums_squares += np.sum(img_array ** 2, axis=(0, 1))
                total_pixels += num_pixels
                image_count += 1
                
            except Exception as e:
                print(f"Ошибка: {img_path}, {e}")
                continue
    
    if total_pixels == 0:
        raise ValueError("Нет валидных изображений для вычисления статистик")
    
    mean = channel_sums / total_pixels
    std = np.sqrt(channel_sums_squares / total_pixels - mean ** 2)
    
    print(f"Обработано изображений: {image_count}")
    print(f"Изображений с низким цветовым разнообразием: {low_diversity_images}")
    print(f"Общее количество пикселей: {total_pixels}")
    
    return mean, std

if __name__ == '__main__':
    dataset_path = os.path.join(BASE_DIR, "dataset")
    train_txt = os.path.join(BASE_DIR, "train.txt")
    
    try:
        train_groups = read_groups_from_txt(train_txt)
    except FileNotFoundError as e:
        print(e)
        train_groups = None
    
    # Вычисляем статистику для 224x224 на всех изображениях
    mean, std = compute_dataset_statistics(dataset_path, groups=train_groups, img_size=224)
    print(f"Среднее по каналам RGB (224x224): {mean}")
    print(f"Стандартное отклонение по каналам RGB (224x224): {std}")