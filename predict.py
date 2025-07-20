import os
import torch
import torch.nn as nn
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import shutil
import random

# Путь к папке, где находится скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Конфигурация
CFG = {
    'model_name': 'swin_small_patch4_window7_224',  # Уменьшили модель
    'img_size': 360,  # Уменьшили размер изображения
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
}

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ToTensorV2()
])

class EnhancedAnimeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('RF5/danbooru-pretrained', 'resnet50', pretrained=True)
        
        # Замораживаем весь backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Размораживаем только последний блок (с осторожностью)
        for param in self.backbone[0][7].parameters():
            param.requires_grad = True
        
        self.backbone[1] = nn.Sequential(
            self.backbone[1][0],  # AdaptiveConcatPool2d
            self.backbone[1][1]   # Flatten
        )
        feature_size = 4096
        
        # Упрощенная голова для большей стабильности
        self.rank_head = nn.Sequential(
            nn.Linear(feature_size, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, x_group):
        batch_size = x_group.size(0)
        features = self.backbone(x_group.flatten(0, 1))
        features = features.view(batch_size*5, -1)
        scores = self.rank_head(features)
        scores = scores.view(batch_size, 5)
        return scores

def predict_auto(model_path, group_folder, temp=1, save_threshold=2.0, complex_save_prob=0.1):
    """
    Автоматически размечает группу и сохраняет в retrain_dir, если confidence_ratio >= save_threshold.
    
    Args:
        model_path (str): Путь к модели.
        group_folder (str): Путь к папке группы.
        temp (float): Температура для softmax.
        save_threshold (float): Порог confidence_ratio для сохранения группы.
    
    Returns:
        str: Имя лучшего изображения.
    """
    model = EnhancedAnimeRanker().to(CFG['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    images = []
    png_files = sorted([f for f in os.listdir(group_folder) if f.endswith('.png')])
    
    for img_file in png_files:
        img_path = os.path.join(group_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        img = val_transform(image=np.array(img))['image']
        images.append(img)
    
    images_tensor = torch.stack(images).unsqueeze(0).to(CFG['device'])
    
    with torch.no_grad():
        scores = model(images_tensor)
    
    scaled_scores = scores / temp
    probs = torch.softmax(scaled_scores, dim=1).cpu().numpy()[0]
    sorted_probs = np.sort(probs)[::-1]
    confidence_ratio = sorted_probs[0] / sorted_probs[1]
    
    # Логирование confidence_ratio
    log_file = "confidence_ratios.txt"
    with open(log_file, "a") as f:
        f.write(f"{confidence_ratio:.4f}\n")
    
    print(f"Sorted probabilities: {sorted_probs}")
    print(f"Confidence ratio: {confidence_ratio:.4f}")
    
    best_idx = torch.argmax(scores).item()
    best_file = png_files[best_idx]
    
    if confidence_ratio >= save_threshold or (confidence_ratio >= 1.2 and random.random() < complex_save_prob):
        save_for_retraining(group_folder, best_file)
    
    return best_file

def save_for_retraining(group_folder, best_file):
    """
    Сохраняет группу изображений в retrain_dir в новую папку с уникальным именем.
    
    Args:
        group_folder (str): Путь к папке с изображениями (например, '/path/to/input').
        best_file (str): Имя лучшего изображения (например, 'image_1.png').
    """
    try:
        # Проверяем CFG['retrain_dir']
        if not CFG.get('retrain_dir'):
            raise ValueError("CFG['retrain_dir'] не задан")
        if not os.path.exists(CFG['retrain_dir']):
            os.makedirs(CFG['retrain_dir'], exist_ok=True)

        # Проверяем group_folder
        if not os.path.isdir(group_folder):
            raise ValueError(f"group_folder не является папкой или не существует: {group_folder}")

        # Генерируем уникальный group_id
        existing_groups = [d for d in os.listdir(CFG['retrain_dir']) if d.startswith('group_') and d[6:].isdigit()]
        if existing_groups:
            max_id = max(int(d[6:]) for d in existing_groups)
            group_id = f"group_{max_id + 1:03d}"
        else:
            group_id = "group_001"
        
        # Формируем dest_folder
        dest_folder = os.path.join(CFG['retrain_dir'], group_id)

        # Проверяем, существует ли dest_folder
        if os.path.exists(dest_folder):
            return

        # Создаём dest_folder
        os.makedirs(dest_folder, exist_ok=True)

        # Проверяем best_file
        png_files = [f for f in os.listdir(group_folder) if f.endswith('.png')]
        if best_file not in png_files:
            raise ValueError(f"best_file '{best_file}' не найден в {group_folder}: {png_files}")

        # Копируем .png файлы
        for img_file in png_files:
            src = os.path.join(group_folder, img_file)
            dst = os.path.join(dest_folder, img_file)
            shutil.copy2(src, dst)

        # Создаём best.txt
        best_txt_path = os.path.join(dest_folder, 'best.txt')
        with open(best_txt_path, 'w') as f:
            f.write(best_file)

    except PermissionError as e:
        raise
    except OSError as e:
        raise
    except ValueError as e:
        raise
    except Exception as e:
        raise

if __name__ == '__main__':
    test_group = os.path.join(BASE_DIR, "input")
    best_image = predict_auto('7244 4259.pth', test_group)
    print(f"\n🎯 Лучшее изображение: {best_image}")