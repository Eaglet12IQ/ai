import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import umap
import pandas as pd
import albumentations as A

# ====================== НАСТРОЙКИ ======================
CFG = {
    'dataset_path': "dataset",                    # путь к папке с группами
    'img_size': 360,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
    'random_state': 42,
}

# ====================== МОДЕЛЬ ======================
class FeatureExtractor(torch.nn.Module):
    def __init__(self, weights_path="resnet50danbooru.pth"):
        super().__init__()
        from danbooru_resnet import resnet50 as danbooru_resnet50
        
        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)
        state_dict = torch.load(weights_path, map_location=CFG['device'], weights_only=True)
        self.backbone.load_state_dict(state_dict)
        
        # Берём только backbone до head (4096 фич)
        self.backbone = torch.nn.Sequential(
            self.backbone[0],           # body
            self.backbone[1][0],        # AdaptiveConcatPool2d
            self.backbone[1][1]         # Flatten
        )
        self.backbone.eval()
        self.backbone.to(CFG['device'])

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)

# ====================== ЗАГРУЗКА ДАННЫХ ======================
def load_all_images_with_labels(root_dir):
    data = []
    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                    std=[0.32821858, 0.31147094, 0.30761928]),
        A.pytorch.ToTensorV2()
    ])
    
    for group in tqdm(os.listdir(root_dir), desc="Сканирование групп"):
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
            
        png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
        if len(png_files) != 5:
            continue
            
        best_txt = os.path.join(group_path, "best.txt")
        if not os.path.exists(best_txt):
            continue
            
        with open(best_txt, "r") as f:
            best_file = f.read().strip()
        
        best_idx = png_files.index(best_file) if best_file in png_files else -1
        
        for idx, filename in enumerate(png_files):
            img_path = os.path.join(group_path, filename)
            is_best = (idx == best_idx)
            
            data.append({
                'group': group,
                'filename': filename,
                'img_path': img_path,
                'is_best': is_best,
                'position': idx
            })
    
    return pd.DataFrame(data)

# ====================== ОСНОВНОЙ СКРИПТ ======================
if __name__ == "__main__":
    print("Загрузка датасета...")
    df = load_all_images_with_labels(CFG['dataset_path'])
    print(f"Найдено изображений: {len(df)} | Из них лучших: {df['is_best'].sum()}")
    
    # Загружаем экстрактор фич
    print("Загрузка backbone модели...")
    extractor = FeatureExtractor()
    
    # Извлекаем эмбеддинги
    embeddings = []
    labels = []      # 1 = best, 0 = other
    groups = []
    positions = []
    
    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                    std=[0.32821858, 0.31147094, 0.30761928]),
        A.pytorch.ToTensorV2()
    ])
    
    print("Извлечение эмбеддингов...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение фич"):
        img = Image.open(row['img_path']).convert('RGB')
        img = np.array(img)
        img_tensor = transform(image=img)['image'].unsqueeze(0).to(CFG['device'])
        
        with torch.no_grad():
            emb = extractor(img_tensor).cpu().numpy().squeeze()
        
        embeddings.append(emb)
        labels.append(1 if row['is_best'] else 0)
        groups.append(row['group'])
        positions.append(row['position'])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Размер эмбеддингов: {embeddings.shape}")
    
    # ====================== UMAP ======================
    print("Запуск UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=CFG['random_state'],
        n_jobs=-1
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    # ====================== ВИЗУАЛИЗАЦИЯ ======================
    plt.figure(figsize=(14, 10))
    
    # Синие — обычные, красные — лучшие
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1],
        c=['#d62728' if is_best else '#1f77b4' for is_best in labels],
        s=12,
        alpha=0.7,
        edgecolors='none'
    )
    
    plt.title('UMAP проекция эмбеддингов изображений\nКрасный = лучшее изображение по твоему вкусу', 
              fontsize=16, pad=20)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Лучшее изображение',
               markerfacecolor='#d62728', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Остальные изображения',
               markerfacecolor='#1f77b4', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем
    plt.savefig('umap_embeddings_anime_ranker.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ====================== ИНТЕРАКТИВНАЯ ВЕРСИЯ (HTML) ======================
    try:
        import plotly.express as px
        
        df_plot = pd.DataFrame({
            'UMAP1': embedding_2d[:, 0],
            'UMAP2': embedding_2d[:, 1],
            'is_best': ['Лучшее' if lb else 'Обычное' for lb in labels],
            'group': groups,
            'position': positions,
            'filename': df['filename'].values
        })
        
        fig = px.scatter(
            df_plot, x='UMAP1', y='UMAP2',
            color='is_best',
            hover_data=['group', 'position', 'filename'],
            color_discrete_map={'Лучшее': '#d62728', 'Обычное': '#1f77b4'},
            title='Интерактивная UMAP проекция эмбеддингов (наведи курсор на точки)',
            width=1000,
            height=800
        )
        fig.write_html("umap_embeddings_interactive.html")
        print("Интерактивная версия сохранена: umap_embeddings_interactive.html")
    except ImportError:
        print("Plotly не установлен — интерактивная версия не создана (pip install plotly)")
    
    print("\nГотово! Файлы сохранены:")
    print("   • umap_embeddings_anime_ranker.png")
    print("   • umap_embeddings_interactive.html (если plotly установлен)")