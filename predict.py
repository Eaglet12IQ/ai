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
import re
import json

# Путь к папке, где находится скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Конфигурация
CFG = {
    'img_size': 224,  # Уменьшили размер изображения
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
}

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.53882785, 0.5202824, 0.51766375], std=[0.33993446, 0.32239504, 0.31772373]),
    ToTensorV2()
])

class EnhancedAnimeRanker(nn.Module):
    def __init__(self):
        super().__init__()

        from danbooru_resnet import resnet50 as danbooru_resnet50

        weights_path = os.path.join(BASE_DIR, "resnet50danbooru.pth")

        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)

        state_dict = torch.load(weights_path, map_location=CFG['device'])
        self.backbone.load_state_dict(state_dict)

        self.backbone = nn.Sequential(
            self.backbone[0],  # body (до head)
            self.backbone[1][0],  # AdaptiveConcatPool2d
            self.backbone[1][1]   # Flatten
        )

        feature_size = 4096  # Размерность признаков для текущего backbone
        
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
    
    def forward(self, x):
        # x имеет форму [batch_size, num_images, 3, 224, 224] или [batch_size, 1, 3, 224, 224]
        batch_size = x.size(0)
        num_images = x.size(1) if x.dim() == 5 else 1
        
        # Преобразуем тензор в [batch_size * num_images, 3, 224, 224]
        x = x.view(-1, 3, x.size(-2), x.size(-1))
        
        features = self.backbone(x)
        
        # Преобразуем признаки обратно в [batch_size, num_images, feature_size]
        features = features.view(batch_size, num_images, -1)
        
        # Применяем rank_head для каждого изображения
        scores = torch.stack([self.rank_head(features[i]) for i in range(batch_size)], dim=0)
        
        return scores.squeeze(-1)  # Удаляем последнюю размерность, возвращаем [batch_size, num_images]

def clean_tag(tag: str) -> str:
    return tag.replace("(", "").replace(")", "").replace("\\", "")

def extract_positive_tags(image_path, description_path=None, max_tags=30):
    """
    Достаёт positive-теги из блока "110" PNG-метаданных и добавляет хеш-теги из последней строки description.
    Пробелы, запятые и тире заменяются на пустоту, повторов нет.
    Теги из description ставятся в начало и обязательно включаются.
    """
    banned = {"gogalking:0.8", "loika:0.15", "33gaff:0.05"}  # уже в очищенном виде

    try:
        img = Image.open(image_path)
        info = img.info
        tags = []

        if "prompt" in info:
            data = json.loads(info["prompt"])

            # Берём теги из блока "110"
            if "110" in data and "inputs" in data["110"] and "positive" in data["110"]["inputs"]:
                positive = data["110"]["inputs"]["positive"]
                raw_tags = positive.split(",")
                tags = []
                for t in raw_tags:
                    t_clean = clean_tag(t).replace(" ", "").replace("-", "")
                    if t_clean and t_clean not in banned:
                        tags.append(t_clean)

        # Берём теги из последней строки description
        desc_tags = []
        if description_path and os.path.exists(description_path):
            with open(description_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    last_line = lines[-2]
                    desc_tags = [clean_tag(t).replace(" ", "").replace("-", "").replace(":", "").lower() for t in last_line.split() if t.startswith("#")]
                    desc_tags = [t[1:] if t.startswith("#") else t for t in desc_tags]

        # Объединяем, убираем дубликаты
        combined_tags, seen = [], set()
        for t in desc_tags + tags:  # description идёт в начало
            if t and t not in seen:
                seen.add(t)
                combined_tags.append(t)

        return combined_tags[:max_tags]

    except Exception as e:
        print(f"⚠️ Ошибка чтения метаданных {image_path}: {e}")
        return None

def predict_auto(model_path, group_folder, saved_images_set, temp=1, save_threshold=2.0, complex_save_prob=0.1):
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
    
    print(f"Вероятности для каждого изображения:")
    for img_file, prob in zip(png_files, probs):
        print(f"{img_file}: {prob*100:.2f}%")
    
    print(f"\nSorted probabilities: {sorted_probs}")
    print(f"Confidence ratio: {confidence_ratio:.4f}")
    
    best_idx = torch.argmax(scores).item()
    best_file = png_files[best_idx]
    
    if confidence_ratio >= save_threshold or (confidence_ratio >= 1.2 and random.random() < complex_save_prob):
        save_for_retraining(group_folder, best_file, saved_images_set)
    
    return best_file

# Множество для хранения уже сохранённых файлов
# Множество для хранения уже сохранённых файлов

def save_for_retraining(group_folder, best_file, saved_images_set):
    """
    Сохраняет группу изображений в retrain_dir в новую папку с уникальным именем,
    только если ВСЕ изображения из группы ещё не были добавлены ранее.
    
    Args:
        group_folder (str): Путь к папке с изображениями
        best_file (str): Имя лучшего изображения
    """

    try:
        if not CFG.get('retrain_dir'):
            raise ValueError("CFG['retrain_dir'] не задан")
        os.makedirs(CFG['retrain_dir'], exist_ok=True)

        if not os.path.isdir(group_folder):
            raise ValueError(f"group_folder не является папкой или не существует: {group_folder}")

        png_files = [f for f in os.listdir(group_folder) if f.endswith('.png')]
        if best_file not in png_files:
            raise ValueError(f"best_file '{best_file}' не найден в {group_folder}: {png_files}")

        # 🔍 Проверка на дубликаты
        if any(img_file in saved_images_set for img_file in png_files):
            print(f"⚠️ Группа пропущена (найден дубликат среди {png_files})")
            return  # просто выходим, не создаём папку

        # Генерируем уникальный group_id
        existing_groups = [d for d in os.listdir(CFG['retrain_dir']) if d.startswith('group_') and d[6:].isdigit()]
        max_id = max([int(d[6:]) for d in existing_groups], default=0)
        group_id = f"group_{max_id + 1:03d}"

        dest_folder = os.path.join(CFG['retrain_dir'], group_id)
        if os.path.exists(dest_folder):
            return
        os.makedirs(dest_folder, exist_ok=True)

        # Копируем все файлы и добавляем их в множество
        for img_file in png_files:
            src = os.path.join(group_folder, img_file)
            dst = os.path.join(dest_folder, img_file)
            shutil.copy2(src, dst)
            saved_images_set.add(img_file)

        # Создаём best.txt
        best_txt_path = os.path.join(dest_folder, 'best.txt')
        with open(best_txt_path, 'w') as f:
            f.write(best_file)

        print(f"✅ Группа сохранена в {dest_folder}")

    except Exception as e:
        raise e

def select_best_4level_flat(
    model_path,
    input_dir,
    group_size=5,
    batch_size=125,
    save_threshold=2.0,
    output_dir=r"D:\finish",
    copy_from_dir=None,
    description=None
):
    import os, shutil
    os.makedirs(output_dir, exist_ok=True)

    if copy_from_dir is None:
        raise ValueError("❌ Укажи copy_from_dir, т.к. копировать надо из другой папки")

    all_pngs = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not all_pngs:
        print("⚠️ Нет .png файлов в папке:", input_dir)
        return []
    
    saved_images_set = set()

    final_results = []
    batches = [all_pngs[i:i+batch_size] for i in range(0, len(all_pngs), batch_size)]

    processed_files = set()  # множество для хранения всех обработанных файлов

    for batch_idx, batch_files in enumerate(batches, 1):
        if len(batch_files) < batch_size:
            print(f"⚠️ Пропускаем неполный пакет ({len(batch_files)} файлов)")
            continue

        if description:
            # Берём первую строку description
            first_line = description.splitlines()[0].strip()
            # Заменяем недопустимые символы для имени папки
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', first_line)
            batch_folder_name = safe_name
        else:
            # fallback на старую логику, если description пустой
            batch_folder_name = f"{batch_files[0].split('.')[0]}-{batch_files[-1].split('.')[0]}"
        batch_folder = os.path.join(output_dir, batch_folder_name)
        os.makedirs(batch_folder, exist_ok=True)

        # ---------- Уровень 1 ----------
        level1_folder = os.path.join(batch_folder, "level1")
        os.makedirs(level1_folder, exist_ok=True)
        level1_best = []

        groups_lvl1 = [batch_files[i:i+group_size] for i in range(0, len(batch_files), group_size)]
        for group in groups_lvl1:
            if len(group) < group_size:
                continue

            temp_folder = os.path.join(batch_folder, "temp_level1")
            os.makedirs(temp_folder, exist_ok=True)

            for img in group:
                shutil.copy2(os.path.join(input_dir, img), os.path.join(temp_folder, img))

            best_img = predict_auto(model_path, temp_folder, saved_images_set, save_threshold=save_threshold)
            level1_best.append(best_img)

            for img in group:
                src = os.path.join(copy_from_dir, img)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(level1_folder, img))

            processed_files.update(group)
            shutil.rmtree(temp_folder, ignore_errors=True)

        # ---------- Уровень 2 ----------
        level2_folder = os.path.join(batch_folder, "level2")
        os.makedirs(level2_folder, exist_ok=True)
        level2_best = []

        groups_lvl2 = [level1_best[i:i+group_size] for i in range(0, len(level1_best), group_size)]
        for group in groups_lvl2:
            if len(group) < group_size:
                continue
            temp_folder = os.path.join(batch_folder, "temp_level2")
            os.makedirs(temp_folder, exist_ok=True)

            for img in group:
                shutil.copy2(os.path.join(input_dir, img), os.path.join(temp_folder, img))

            best_img = predict_auto(model_path, temp_folder, saved_images_set, save_threshold=save_threshold)
            level2_best.append(best_img)

            for img in group:
                src = os.path.join(copy_from_dir, img)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(level2_folder, img))

            processed_files.update(group)
            shutil.rmtree(temp_folder, ignore_errors=True)

        # ---------- Уровень 3 ----------
        level3_folder = os.path.join(batch_folder, "level3")
        os.makedirs(level3_folder, exist_ok=True)
        level3_best = []

        groups_lvl3 = [level2_best[i:i+group_size] for i in range(0, len(level2_best), group_size)]
        for group in groups_lvl3:
            if len(group) < group_size:
                continue
            temp_folder = os.path.join(batch_folder, "temp_level3")
            os.makedirs(temp_folder, exist_ok=True)

            for img in group:
                shutil.copy2(os.path.join(input_dir, img), os.path.join(temp_folder, img))

            best_img = predict_auto(model_path, temp_folder, saved_images_set, save_threshold=save_threshold)
            level3_best.append(best_img)

            for img in group:
                src = os.path.join(copy_from_dir, img)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(level3_folder, img))

            processed_files.update(group)
            shutil.rmtree(temp_folder, ignore_errors=True)

        # ---------- Финал ----------
        if not level3_best:
            print(f"⚠️ Не удалось выбрать лучшее изображение в пакете {batch_idx}")
            continue

        final_best = level3_best[0]
        final_results.append(final_best)

        src_final = os.path.join(copy_from_dir, final_best)
        dst_final = os.path.join(batch_folder, final_best)
        if os.path.exists(src_final):
            shutil.copy2(src_final, dst_final)

        if description:
            desc_path = os.path.join(batch_folder, "description.txt")
            with open(desc_path, "w", encoding="utf-8") as f:
                f.write(str(description))

        tags = extract_positive_tags(src_final, desc_path, max_tags=30)
        if tags:
            with open(desc_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n" + " ".join(tags))

        for root, _, files in os.walk(batch_folder):
            for img_file in files:
                if img_file.endswith(".png"):
                    img_path = os.path.join(root, img_file)
                    try:
                        with Image.open(img_path) as img:
                            data = np.array(img)
                        img_clean = Image.fromarray(data)
                        img_clean.save(img_path)  # теперь работает
                    except Exception as e:
                        print(f"⚠️ Ошибка очистки метаданных {img_file}: {e}")

        archive_name = os.path.join(batch_folder, os.path.basename(batch_folder))  # полный путь + имя батча
        shutil.make_archive(archive_name, 'zip', level1_folder)

    # ---------- Удаляем все обработанные файлы в самом конце ----------
    for img in processed_files:
        for dir_path in [input_dir, copy_from_dir]:
            try:
                os.remove(os.path.join(dir_path, img))
            except FileNotFoundError:
                pass

    return final_results

if __name__ == '__main__':
    input_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUI\output\dataset"
    model_path = "7293 4592 best.pth"
    output_dir = r"D:\finish"
    copy_from_dir = r"D:\StabilityMatrix\Data\Packages\ComfyUI\output"

    select_best_4level_flat(model_path, input_dir, group_size=5, batch_size=125, save_threshold=2.0, output_dir=output_dir, copy_from_dir=copy_from_dir)