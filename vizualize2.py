import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# ====================== НАСТРОЙКИ ======================
CFG = {
    'dataset_path': "dataset",
    'img_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_state': 42,
    'worst_save_count': 10,                    # сколько худших сохранять
    'worst_dir': 'worst_augmentations',
}

# ====================== МОДЕЛЬ ======================
class FeatureExtractor(torch.nn.Module):
    def __init__(self, weights_path="resnet50danbooru.pth"):
        super().__init__()
        from danbooru_resnet import resnet50 as danbooru_resnet50
        
        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)
        state_dict = torch.load(weights_path, map_location=CFG['device'], weights_only=True)
        self.backbone.load_state_dict(state_dict)
        
        self.backbone = torch.nn.Sequential(
            self.backbone[0],
            self.backbone[1][0],
            self.backbone[1][1]
        )
        self.backbone.eval()
        self.backbone.to(CFG['device'])

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)


# ====================== ТРАНСФОРМЫ ======================
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=360),
    A.CenterCrop(360, 360),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
])

train_transform = A.Compose([
    A.RandomResizedCrop(size=(CFG['img_size'], CFG['img_size']), scale=(0.85, 1.0), ratio=(0.75, 1.35)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.07),
        hole_width_range=(0.03, 0.07),
        fill=(0.54288839*255, 0.52424041*255, 0.52013308*255),
        p=0.4                     # ← Увеличено
    ),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308], std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
], seed=CFG['random_state'])


def get_all_images_from_all_groups(root_dir):
    image_paths = []
    for group in sorted(os.listdir(root_dir)):
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        png_files = [f for f in os.listdir(group_path) if f.endswith('.png')]
        if len(png_files) != 5:
            continue
        for filename in png_files:
            img_path = os.path.join(group_path, filename)
            image_paths.append((group, filename, img_path))
    
    print(f"Найдено групп: {len(set(g for g, _, _ in image_paths))}")
    print(f"Всего изображений для анализа: {len(image_paths)}")
    return image_paths


# ====================== ОСНОВНОЙ СКРИПТ ======================
if __name__ == "__main__":
    print("Загрузка модели...")
    extractor = FeatureExtractor()

    print("Сбор всех изображений из всех групп...")
    image_list = get_all_images_from_all_groups(CFG['dataset_path'])

    similarities = []
    worst_examples = []   # список для хранения худших случаев

    np.random.seed(CFG['random_state'])

    print("Вычисление сходства оригинал ↔ аугментация...")
    for group, filename, img_path in tqdm(image_list, desc="Анализ аугментаций"):
        try:
            orig_img = Image.open(img_path).convert('RGB')
            img_np = np.array(orig_img)
            
            # Оригинал
            orig_tensor = val_transform(image=img_np)['image'].unsqueeze(0).to(CFG['device'])
            orig_emb = extractor(orig_tensor)
            orig_emb = torch.nn.functional.normalize(orig_emb, p=2, dim=1)

            for aug_idx in range(4):   # 4 аугментации на изображение
                aug_tensor = train_transform(image=img_np)['image'].unsqueeze(0).to(CFG['device'])
                aug_emb = extractor(aug_tensor)
                aug_emb = torch.nn.functional.normalize(aug_emb, p=2, dim=1)

                sim = torch.cosine_similarity(orig_emb, aug_emb).item()
                similarities.append(sim)

                # Сохраняем информацию о худших примерах
                if sim < 0.85:   # порог для кандидатов в "худшие"
                    worst_examples.append({
                        'similarity': sim,
                        'group': group,
                        'filename': filename,
                        'img_path': img_path,
                        'aug_idx': aug_idx
                    })

        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            continue

    similarities = np.array(similarities)

    # ====================== СТАТИСТИКА ======================
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА АУГМЕНТАЦИЙ (ПО ВСЕМ ИЗОБРАЖЕНИЯМ)")
    print("="*80)
    print(f"Всего вычислено пар сходства: {len(similarities)}")
    print(f"Среднее косинусное сходство: {similarities.mean():.4f}")
    print(f"Медиана:                    {np.median(similarities):.4f}")
    print(f"Стд. отклонение:            {similarities.std():.4f}")
    print(f"Min сходство:               {similarities.min():.4f}")
    print(f"Max сходство:               {similarities.max():.4f}")
    print(f"Процент < 0.90:             {(similarities < 0.90).mean()*100:.2f}%")
    print(f"Процент < 0.85:             {(similarities < 0.85).mean()*100:.2f}%")
    print(f"Процент < 0.80:             {(similarities < 0.80).mean()*100:.2f}%")

# ====================== ХУДШИЕ АУГМЕНТАЦИИ ======================
    print("\n" + "="*70)
    print("ТОП-10 ХУДШИХ АУГМЕНТАЦИЙ (по косинусному сходству)")
    print("="*70)

    # Сортируем по худшим (самое низкое сходство сначала)
    worst_examples = sorted(worst_examples, key=lambda x: x['similarity'])[:CFG['worst_save_count']]

    worst_dir = Path(CFG['worst_dir'])
    worst_dir.mkdir(exist_ok=True)

    mean = [0.54288839, 0.52424041, 0.52013308]
    std  = [0.32821858, 0.31147094, 0.30761928]

    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    for i, item in enumerate(worst_examples, 1):
        sim = item['similarity']
        orig_pil = Image.open(item['img_path']).convert('RGB')
        img_np = np.array(orig_pil)

        # Применяем аугментацию заново
        aug_result = train_transform(image=img_np)
        aug_tensor = aug_result['image']                    # уже в [C, H, W], normalized

        # Денормализация
        aug_denorm = denormalize(aug_tensor.unsqueeze(0), mean, std)[0]
        aug_denorm = torch.clamp(aug_denorm, 0, 1)

        aug_pil = Image.fromarray(
            (aug_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

        # Создаём side-by-side изображение
        combined = Image.new('RGB', (orig_pil.width * 2 + 20, orig_pil.height))
        combined.paste(orig_pil, (0, 0))
        combined.paste(aug_pil, (orig_pil.width + 20, 0))

        # Добавляем текст
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        text = f"Similarity: {sim:.4f}   |   Group: {item['group']}   |   File: {item['filename']}"
        draw.text((10, 10), text, fill=(255, 255, 0), font=font)

        save_path = worst_dir / f"worst_{i:02d}_sim{sim:.4f}_{item['group']}.jpg"
        combined.save(save_path, quality=95)

        print(f"{i:2d}. Similarity = {sim:.4f}  |  Group: {item['group']}  |  {item['filename']}")

    print(f"\n✅ Худшие аугментации сохранены в папку: ./{CFG['worst_dir']}/")
    print("   (слева — оригинал, справа — аугментированная версия)")

    # ====================== ВИЗУАЛИЗАЦИЯ ======================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    sns.histplot(similarities, bins=80, kde=True, ax=axs[0, 0], color='#1f77b4')
    axs[0, 0].axvline(similarities.mean(), color='red', linestyle='--', label=f'Среднее = {similarities.mean():.4f}')
    axs[0, 0].set_title('Распределение косинусного сходства')
    axs[0, 0].set_xlabel('Косинусное сходство')
    axs[0, 0].set_ylabel('Частота')
    axs[0, 0].legend()

    sns.violinplot(y=similarities, ax=axs[0, 1], color='#ff7f0e')
    sns.boxplot(y=similarities, ax=axs[0, 1], width=0.25, color='white')
    axs[0, 1].set_title('Box + Violin plot')
    axs[0, 1].set_ylabel('Косинусное сходство')

    sns.ecdfplot(similarities, ax=axs[1, 0], color='#2ca02c')
    axs[1, 0].set_title('Кумулятивное распределение')
    axs[1, 0].set_xlabel('Косинусное сходство')
    axs[1, 0].set_ylabel('Доля ≤ x')
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].axis('off')
    stats_text = f"""Полная статистика:
Всего пар:         {len(similarities)}
Среднее:           {similarities.mean():.4f}
Медиана:           {np.median(similarities):.4f}
Минимум:           {similarities.min():.4f}
< 0.90 : {(similarities < 0.90).mean()*100:.1f}%
< 0.85 : {(similarities < 0.85).mean()*100:.1f}%
< 0.80 : {(similarities < 0.80).mean()*100:.1f}%
"""
    axs[1, 1].text(0.05, 0.5, stats_text, fontsize=12, va='center', fontfamily='monospace')

    plt.suptitle('Анализ устойчивости эмбеддингов к аугментациям\n(все 1110 групп × все изображения)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('augmentation_robustness_full_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    pd.DataFrame({'cosine_similarity': similarities}).to_csv('augmentation_similarities_full.csv', index=False)

    print("\nГотово!")
    print("График сохранён → augmentation_robustness_full_analysis.png")
    print("Данные сохранены → augmentation_similarities_full.csv")