import os
import random
import torch
import torch.nn as nn
import torch.hub
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler

# Фиксация генераторов случайных чисел
seed = 49
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
plt.rcParams['font.family'] = 'Segoe UI Emoji'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_valid_groups(root_dir):
    valid_groups = []
    for group in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        png_files = [f for f in os.listdir(group_path) if f.endswith('.png')]
        best_txt = os.path.join(group_path, "best.txt")
        if len(png_files) == 5 and os.path.exists(best_txt):
            with open(best_txt, "r") as f:
                best_file = f.read().strip()
            if best_file in png_files:
                valid_groups.append(group)
    return valid_groups

def my_collate_fn(batch):
    best_images = torch.stack([item[0] for item in batch])
    other_images = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    group_names = [item[3] for item in batch]
    return best_images, other_images, targets, group_names

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

CFG = {
    'img_size': 360,
    'batch_size': 32,
    'num_workers': 6,
    'grad_clip': 1.0,
    'weight_decay': 0.01,
    'head_lr': 1e-4,
    'backbone_lr': 5e-6,          # ← Увеличено (было 1e-6)
    'min_lr': 5e-7,
    'epochs': 150,
    'swa_start': 60,              # ← Раньше стартуем SWA
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
    'dataset_path': os.path.join(BASE_DIR, "dataset")
}

# Аугментации
train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.75, 1.35)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.07),
        hole_width_range=(0.03, 0.07),
        fill=(0.54288839*255, 0.52424041*255, 0.52013308*255),
        p=0.4                     # ← Увеличено
    ),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308], std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
], seed=seed)

# Исправленный val_transform (убран баг с искажением aspect ratio)
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=CFG['img_size']),   # сохраняем пропорции
    A.CenterCrop(CFG['img_size'], CFG['img_size']),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
])

class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, groups=None):
        self.root_dir = root_dir
        self.transform = transform
        all_groups = groups or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.groups = []
        self.pairs = []
       
        for group in all_groups:
            group_path = os.path.join(root_dir, group)
            png_files = [f for f in os.listdir(group_path) if f.endswith('.png')]
            if len(png_files) != 5:
                continue
            best_txt = os.path.join(group_path, 'best.txt')
            if os.path.exists(best_txt):
                with open(best_txt, 'r') as f:
                    best_file = f.read().strip()
                if best_file in png_files:
                    self.groups.append(group)
                    best_idx = png_files.index(best_file)
                    for other_idx in range(5):
                        if other_idx != best_idx:
                            self.pairs.append((group, best_idx, other_idx, 1))
                            self.pairs.append((group, other_idx, best_idx, -1))
                else:
                    print(f"Предупреждение: в группе {group} файл best.txt указывает на неверный файл {best_file}")
            else:
                print(f"Предупреждение: файл best.txt не найден в группе {group}")

    def __len__(self):
        return len(self.pairs)
   
    def __getitem__(self, idx):
        group, best_idx, other_idx, target = self.pairs[idx]
        group_path = os.path.join(self.root_dir, group)
        png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
       
        best_img_path = os.path.join(group_path, png_files[best_idx])
        best_img = Image.open(best_img_path).convert('RGB')
        best_img = np.array(best_img)
        if np.any(np.isnan(best_img)) or np.any(np.isinf(best_img)):
            print(f"Предупреждение: NaN или Inf в лучшем изображении {best_img_path}")
            best_img = np.zeros_like(best_img)
       
        other_img_path = os.path.join(group_path, png_files[other_idx])
        other_img = Image.open(other_img_path).convert('RGB')
        other_img = np.array(other_img)
        if np.any(np.isnan(other_img)) or np.any(np.isinf(other_img)):
            print(f"Предупреждение: NaN или Inf в другом изображении {other_img_path}")
            other_img = np.zeros_like(other_img)
       
        if self.transform:
            best_img = self.transform(image=best_img)['image']
            other_img = self.transform(image=other_img)['image']
       
        return best_img, other_img, torch.tensor(target, dtype=torch.float32), group

class EnhancedAnimeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        from danbooru_resnet import resnet50 as danbooru_resnet50
        weights_path = os.path.join(BASE_DIR, "resnet50danbooru.pth")
        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)
        state_dict = torch.load(weights_path, map_location=CFG['device'])
        self.backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(
            self.backbone[0],  # body
            self.backbone[1][0],  # AdaptiveConcatPool2d
            self.backbone[1][1]   # Flatten
        )
        feature_size = 4096
       
        # Упрощённая и более стабильная голова
        self.rank_head = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
   
    def forward(self, x):
        """Векторизованный forward — убрали медленный Python-цикл"""
        batch_size = x.size(0)
        num_images = x.size(1) if x.dim() == 5 else 1
        
        # Приводим к (B * num_images, 3, H, W)
        x = x.view(-1, 3, x.size(-2), x.size(-1))
        features = self.backbone(x)                    # (B*num, 4096)
        features = features.view(batch_size, num_images, -1)
        
        # Векторизованный проход через голову
        flat_features = features.view(batch_size * num_images, -1)
        flat_scores = self.rank_head(flat_features)    # (B*num, 1)
        scores = flat_scores.view(batch_size, num_images)
        
        # Совместимость с оригинальным кодом (при num_images=1 возвращаем (B,))
        return scores.squeeze(-1) if num_images == 1 else scores

class SoftFocalPairwiseLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, margin=0.9, temperature=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.temperature = temperature
   
    def forward(self, scores_best, scores_other, target):
        diff = target * (scores_best - scores_other) / self.temperature
        adjusted = self.margin - diff
       
        pt = torch.sigmoid(diff)
        focal_weight = (1 - pt).pow(self.gamma)
       
        loss = self.alpha * focal_weight * torch.log1p(torch.exp(adjusted))
        return loss.mean()

def read_groups_from_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, "r") as f:
        groups = [line.strip() for line in f.readlines() if line.strip()]
    return groups

def evaluate(model, dataloader, device):
    model.eval()
    top1_correct = 0
    top2_correct = 0
    ndcg_scores = []
    val_loss_sum = 0.0
    val_loss_count = 0
    processed_groups = 0
   
    criterion = SoftFocalPairwiseLoss()
   
    with torch.no_grad():
        group_images_dict = {}
       
        for best_images, other_images, targets, group_names in dataloader:
            best_images = best_images.to(device, non_blocking=True)
            other_images = other_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
           
            scores_best = model(best_images.unsqueeze(1)).squeeze(-1)
            scores_other = model(other_images.unsqueeze(1)).squeeze(-1)
           
            batch_loss = criterion(scores_best, scores_other, targets)
            val_loss_sum += batch_loss.item() * best_images.size(0)
            val_loss_count += best_images.size(0)
           
            for i in range(best_images.size(0)):
                group_id = group_names[i]
                if group_id not in group_images_dict:
                    group_images_dict[group_id] = []
                group_images_dict[group_id].append(best_images[i:i+1].cpu())
                group_images_dict[group_id].append(other_images[i:i+1].cpu())
       
        for group_id, images_list in group_images_dict.items():
            if len(images_list) < 8:
                continue
           
            unique_imgs = []
            seen = set()
            for img_tensor in images_list:
                img_hash = hash(img_tensor.cpu().numpy().tobytes())
                if img_hash not in seen:
                    seen.add(img_hash)
                    unique_imgs.append(img_tensor)
           
            if len(unique_imgs) != 5:
                continue
           
            group_tensor = torch.cat(unique_imgs[:5], dim=0).to(device)
            scores = model(group_tensor.unsqueeze(1)).squeeze(-1)
           
            true_best_idx = 0
            ranked_indices = torch.argsort(scores, dim=0, descending=True)
           
            if ranked_indices[0] == true_best_idx:
                top1_correct += 1
            if true_best_idx in ranked_indices[:2]:
                top2_correct += 1
           
            true_relevance = np.zeros((1, 5), dtype=np.float32)
            true_relevance[0, true_best_idx] = 1.0
           
            try:
                batch_ndcg = ndcg_score(true_relevance, scores.unsqueeze(0).cpu().numpy(), k=5)
                ndcg_scores.append(batch_ndcg)
            except Exception as e:
                print(f"Ошибка в NDCG для группы {group_id}: {e}")
           
            processed_groups += 1
   
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_val_loss = val_loss_sum / val_loss_count if val_loss_count > 0 else 0.0
   
    return {
        'top1': top1_correct / processed_groups if processed_groups > 0 else 0.0,
        'top2': top2_correct / processed_groups if processed_groups > 0 else 0.0,
        'ndcg': mean_ndcg,
        'val_loss': avg_val_loss,
        'groups': processed_groups
    }

def train():
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)
   
    # Загружаем группы из txt
    train_txt = os.path.join(BASE_DIR, "train.txt")
    val_txt = os.path.join(BASE_DIR, "val.txt")
   
    train_groups = read_groups_from_txt(train_txt)
    val_groups = read_groups_from_txt(val_txt)
   
    train_ds = AnimeGroupDataset(CFG['dataset_path'], transform=train_transform, groups=train_groups)
    val_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=val_groups)
    retrain_ds = AnimeGroupDataset(CFG['retrain_dir'], transform=train_transform)
   
    # ← Добавляем retrain данные (очень важно!)
    # if len(retrain_ds) > 0:
    #     train_ds = ConcatDataset([train_ds, retrain_ds])
    #     print(f"Добавлено {len(retrain_ds)} пар из retrain-директории")
   
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        persistent_workers=True,
    )
   
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
   
    model = EnhancedAnimeRanker().to(CFG['device'])
   
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG['backbone_lr']},
        {'params': model.rank_head.parameters(), 'lr': CFG['head_lr']}
    ], weight_decay=CFG['weight_decay'])
   
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-7)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=CFG['min_lr'])
    criterion = SoftFocalPairwiseLoss()
    scaler = GradScaler('cuda')          # ← AMP включён
   
    best_ndcg = 0
    best_top1 = 0
    best_swa_ndcg = 0
    patience = 20
    epochs_without_improvement = 0
   
    history = {
        'train_loss': [],
        'val_ndcg': [],
        'val_top1': [],
        'val_loss': [],
        'swa_val_ndcg': [],
        'swa_val_top1': []
    }
   
    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0
        epoch_head_grad_norm = 0
        epoch_backbone_grad_norm = 0
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1}/{CFG["epochs"]}')
       
        for batch_idx, (best_images, other_images, targets, _) in enumerate(progress_bar):
            best_images = best_images.to(CFG['device'], non_blocking=True)
            other_images = other_images.to(CFG['device'], non_blocking=True)
            targets = targets.to(CFG['device'], non_blocking=True)
           
            scores_best = model(best_images.unsqueeze(1))
            scores_other = model(other_images.unsqueeze(1))
           
            loss = criterion(scores_best, scores_other, targets)
           
            # === AMP + корректный gradient clipping ===
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # нужно для правильного клиппинга
           
            head_grad_norm = torch.nn.utils.clip_grad_norm_(model.rank_head.parameters(), CFG['grad_clip'])
            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), CFG['grad_clip'])
           
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
           
            epoch_loss += loss.item()
            epoch_head_grad_norm += head_grad_norm.item()
            epoch_backbone_grad_norm += backbone_grad_norm.item()
           
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
       
        avg_head_grad_norm = epoch_head_grad_norm / len(train_loader)
        avg_backbone_grad_norm = epoch_backbone_grad_norm / len(train_loader)
       
        val_metrics = evaluate(model, val_loader, CFG['device'])
       
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_ndcg'].append(val_metrics['ndcg'])
        history['val_top1'].append(val_metrics['top1'])
        history['val_loss'].append(val_metrics['val_loss'])
       
        # SWA
        if epoch >= CFG['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_bn(train_loader, swa_model, device=CFG['device'])
            swa_metrics = evaluate(swa_model, val_loader, CFG['device'])
            history['swa_val_ndcg'].append(swa_metrics['ndcg'])
            history['swa_val_top1'].append(swa_metrics['top1'])
            print(f"SWA Val NDCG: {swa_metrics['ndcg']:.4f}, SWA Val Top-1: {swa_metrics['top1']:.4f}")
            if swa_metrics['ndcg'] > best_swa_ndcg:
                best_swa_ndcg = swa_metrics['ndcg']
                torch.save(swa_model.module.state_dict(), 'best_swa_model.pth')
                print(f"Новая лучшая SWA модель сохранена с NDCG: {best_swa_ndcg:.4f}")
        else:
            history['swa_val_ndcg'].append(0.0)
            history['swa_val_top1'].append(0.0)
       
        scheduler.step()
       
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            best_top1 = val_metrics['top1']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Новая лучшая модель сохранена с NDCG: {best_ndcg:.4f}, Top-1: {best_top1:.4f}")
        else:
            epochs_without_improvement += 1
       
        if epochs_without_improvement >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break
       
        print(f"\nЭпоха {epoch+1} Итоги:")
        print(f"Потери на обучении: {history['train_loss'][-1]:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val NDCG: {val_metrics['ndcg']:.4f}")
        print(f"Val Top-1 Точность: {val_metrics['top1']:.4f}")
        print(f"Val Top-2 Точность: {val_metrics['top2']:.4f}")
        print(f"Средняя норма градиента головы: {avg_head_grad_norm:.4f}")
        print(f"Средняя норма градиента backbone: {avg_backbone_grad_norm:.4f}")
        print("-" * 50)
   
    torch.save(model.state_dict(), 'final_model.pth')
    model.load_state_dict(torch.load('best_model.pth', map_location=CFG['device']))
    print("Обучение завершено!")

if __name__ == '__main__':
    train()