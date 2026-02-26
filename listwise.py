import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler

# ==================== ФИКСАЦИЯ СИДОВ ====================
seed = 49
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
plt.rcParams['font.family'] = 'Segoe UI Emoji'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    'img_size': 224,
    'batch_size': 32,           # уменьшил из-за (B, 5) изображений
    'num_workers': 8,
    'grad_clip': 1.0,
    'weight_decay': 0.01,
    'head_lr': 1e-4,
    'backbone_lr': 1e-6,
    'min_lr': 5e-7,
    'epochs': 150,
    'swa_start': 80,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
    'dataset_path': os.path.join(BASE_DIR, "dataset"),
    'lr_patience': 5,
    'lr_factor': 0.5,
    'lr_threshold': 1e-4,
    'temperature': 0.72,        # очень важный гиперпараметр!
}

# ==================== АУГМЕНТАЦИИ ====================
train_transform = A.Compose([
    A.Resize(224, 224),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.03, 0.07),
                    hole_width_range=(0.03, 0.07), fill=(0.543*255, 0.525*255, 0.522*255), p=0.3),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308], std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
], seed=seed)

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308], std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
], seed=seed)

# ==================== LISTWISE DATASET ====================
class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, groups=None):
        self.root_dir = root_dir
        self.transform = transform
        all_groups = groups or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.data = []
        
        for group in all_groups:
            group_path = os.path.join(root_dir, group)
            png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
            if len(png_files) != 5:
                continue
                
            best_txt = os.path.join(group_path, 'best.txt')
            if os.path.exists(best_txt):
                with open(best_txt, 'r') as f:
                    best_file = f.read().strip()
                if best_file in png_files:
                    best_idx = png_files.index(best_file)
                    self.data.append((group, best_idx, png_files))
                else:
                    print(f"Предупреждение: неверный best.txt в группе {group}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        group, best_idx, png_files = self.data[idx]
        group_path = os.path.join(self.root_dir, group)
        
        images = []
        for fname in png_files:
            img_path = os.path.join(group_path, fname)
            img = np.array(Image.open(img_path).convert('RGB'))
            
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                img = np.zeros_like(img)
            
            if self.transform:
                img = self.transform(image=img)['image']
            images.append(img)
        
        return torch.stack(images), torch.tensor(best_idx, dtype=torch.long), group


def my_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])        # (B, 5, 3, H, W)
    best_indices = torch.stack([item[1] for item in batch])  # (B,)
    group_names = [item[2] for item in batch]
    return images, best_indices, group_names


# ==================== МОДЕЛЬ ====================
class EnhancedAnimeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        from danbooru_resnet import resnet50 as danbooru_resnet50
        weights_path = os.path.join(BASE_DIR, "resnet50danbooru.pth")
        
        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)
        state_dict = torch.load(weights_path, map_location=CFG['device'])
        self.backbone.load_state_dict(state_dict)
        
        self.backbone = nn.Sequential(
            self.backbone[0],
            self.backbone[1][0],
            self.backbone[1][1]
        )
        
        feature_size = 4096
        self.rank_head = nn.Sequential(
            nn.Linear(feature_size, 2048), nn.GELU(), nn.LayerNorm(2048), nn.Dropout(0.3),
            nn.Linear(2048, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(0.2),
            nn.Linear(1024, 512),  nn.GELU(), nn.LayerNorm(512),  nn.Dropout(0.2),
            nn.Linear(512, 256),   nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """x: (B, 5, 3, H, W) → scores: (B, 5)"""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)               # (B*5, 3, H, W)
        features = self.backbone(x)              # (B*5, 4096)
        scores = self.rank_head(features)        # (B*5, 1)
        scores = scores.view(B, N)               # (B, 5)
        return scores


# ==================== LISTWISE LOSS (твой текущий) ====================
class ListwiseLoss(nn.Module):
    def __init__(self, temperature=0.72, label_smoothing=0.05):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, scores, target):
        scaled = scores / self.temperature
        log_probs = F.log_softmax(scaled, dim=1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / 4.0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        
        loss = -(true_dist * log_probs).sum(dim=1).mean()
        return loss

# ==================== PAIRWISE LOSS (новое) ====================
class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, scores, target):
        """
        scores:  (B, 5)
        target:  (B,) — индекс лучшего изображения
        """
        B, N = scores.shape
        total_loss = 0.0
        count = 0

        for b in range(B):
            best_idx = target[b]
            s_best = scores[b, best_idx]
            
            for i in range(N):
                if i == best_idx:
                    continue
                diff = s_best - scores[b, i]
                # hinge loss: max(0, margin - (s_best - s_other))
                total_loss += torch.relu(self.margin - diff)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        return total_loss / count

# ==================== EVALUATE ====================
def evaluate(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []          # индексы лучшего изображения
    
    with torch.no_grad():
        for images, best_indices, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            scores = model(images)                    # (B, 5)
            
            all_scores.append(scores.cpu())           # сразу на CPU
            all_labels.append(best_indices.cpu())
    
    # Объединяем все батчи
    all_scores = torch.cat(all_scores, dim=0).numpy()   # (N_groups, 5)
    all_labels = torch.cat(all_labels, dim=0).numpy()   # (N_groups,)
    
    # Ground truth в формате one-hot
    batch_true = np.zeros_like(all_scores, dtype=np.float32)
    batch_true[np.arange(len(all_labels)), all_labels] = 1.0
    
    # Один большой NDCG по всему валидационному набору
    ndcg = ndcg_score(batch_true, all_scores, k=5)
    
    # Top-1 и Top-2
    ranked = np.argsort(-all_scores, axis=1)  # descending
    top1 = (ranked[:, 0] == all_labels).mean()
    top2 = np.any(ranked[:, :2] == all_labels[:, None], axis=1).mean()
    
    return {
        'top1': float(top1),
        'top2': float(top2),
        'ndcg': float(ndcg)
    }

def read_groups_from_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ==================== TRAIN ====================
def train():
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)
    
    train_groups = read_groups_from_txt(os.path.join(BASE_DIR, "train.txt"))
    val_groups = read_groups_from_txt(os.path.join(BASE_DIR, "val.txt"))
    
    train_ds = AnimeGroupDataset(CFG['dataset_path'], transform=train_transform, groups=train_groups)
    val_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=val_groups)
    retrain_ds = AnimeGroupDataset(CFG['retrain_dir'], transform=train_transform)
    
    train_loader = DataLoader(
        ConcatDataset([train_ds, retrain_ds]),
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    model = EnhancedAnimeRanker().to(CFG['device'])
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG['backbone_lr']},
        {'params': model.rank_head.parameters(), 'lr': CFG['head_lr']}
    ], weight_decay=CFG['weight_decay'])
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-7)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=CFG['lr_factor'],
                                  patience=CFG['lr_patience'], threshold=CFG['lr_threshold'],
                                  min_lr=CFG['min_lr'])
    
    # ──── НОВОЕ ────
    listwise_criterion = ListwiseLoss(temperature=CFG['temperature'], label_smoothing=0.05)
    pairwise_criterion = PairwiseMarginLoss(margin=0.4)          # ← можно подбирать 0.2–0.7
    
    # Веса компонент лосса (очень важные гиперпараметры!)
    LISTWISE_WEIGHT = 0.70
    PAIRWISE_WEIGHT = 0.30
    
    scaler = GradScaler('cuda')
    
    best_ndcg = 0.0
    patience = 12
    epochs_no_improve = 0
    
    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_listwise = 0.0
        epoch_pairwise = 0.0
        epoch_head_norm = 0.0
        epoch_backbone_norm = 0.0
        
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{CFG['epochs']}")
        
        for images, best_idx, _ in pbar:
            images = images.to(CFG['device'], non_blocking=True)
            best_idx = best_idx.to(CFG['device'], non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda'):
                scores = model(images)                        # (B, 5)
                
                listwise_loss = listwise_criterion(scores, best_idx)
                pairwise_loss = pairwise_criterion(scores, best_idx)
                
                total_loss = (LISTWISE_WEIGHT * listwise_loss +
                              PAIRWISE_WEIGHT * pairwise_loss)
            
            scaler.scale(total_loss).backward()
            
            scaler.unscale_(optimizer)
            
            head_norm = torch.nn.utils.clip_grad_norm_(model.rank_head.parameters(), CFG['grad_clip'])
            backbone_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), CFG['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            epoch_listwise += listwise_loss.item()
            epoch_pairwise += pairwise_loss.item()
            epoch_head_norm += head_norm.item()
            epoch_backbone_norm += backbone_norm.item()
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        avg_listwise = epoch_listwise / len(train_loader)
        avg_pairwise = epoch_pairwise / len(train_loader)
        avg_head = epoch_head_norm / len(train_loader)
        avg_bb = epoch_backbone_norm / len(train_loader)
        
        val_metrics = evaluate(model, val_loader, CFG['device'])
        scheduler.step(val_metrics['ndcg'])
        
        print(f"\nЭпоха {epoch+1} | Total Loss: {avg_loss:.4f} | "
              f"Listwise: {avg_listwise:.4f} | Pairwise: {avg_pairwise:.4f}")
        print(f"Val NDCG: {val_metrics['ndcg']:.4f} | Top-1: {val_metrics['top1']:.4f} | Top-2: {val_metrics['top2']:.4f}")
        print(f"   head_norm: {avg_head:.4f} | bb_norm: {avg_bb:.4f}")
        
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            torch.save(model.state_dict(), 'best_model_multi.pth')
            print(f"НОВАЯ ЛУЧШАЯ МОДЕЛЬ! NDCG = {best_ndcg:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epoch >= CFG['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_bn(train_loader, swa_model, device=CFG['device'])
        
        if epochs_no_improve >= patience:
            print("Ранняя остановка")
            break
    
    torch.save(model.state_dict(), 'final_model_multi.pth')
    print("Обучение завершено!")

if __name__ == '__main__':
    train()