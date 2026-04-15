import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler

# ====================== ФИКСАЦИЯ СИДА ======================
seed = 49
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== КОНФИГУРАЦИЯ ======================
CFG = {
    'img_size': 256,
    'batch_size': 32,
    'num_workers': 6,
    'grad_clip': 1.0,
    'weight_decay': 0.01,
    'head_lr': 1e-4,
    'backbone_lr': 5e-6,
    'min_lr': 5e-7,
    'epochs': 150,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_path': os.path.join(BASE_DIR, "dataset"),
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
    
    # === ФЛАГ ВЫБОРА ДАТАСЕТА ===
    'train_mode': 'synth',   # 'manual' | 'synth' | 'both'
}

# ====================== АУГМЕНТАЦИИ ======================
train_transform = A.Compose([
    A.RandomResizedCrop(size=(CFG['img_size'], CFG['img_size']), scale=(0.85, 1.0), ratio=(0.75, 1.35)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.07),
        hole_width_range=(0.03, 0.07),
        fill=(0.54288839*255, 0.52424041*255, 0.52013308*255),
        p=0.4
    ),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
], seed=seed)

val_transform = A.Compose([
    A.SmallestMaxSize(max_size=360),
    A.CenterCrop(360, 360),
    A.Normalize(mean=[0.54288839, 0.52424041, 0.52013308],
                std=[0.32821858, 0.31147094, 0.30761928]),
    ToTensorV2()
])


def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)


# ====================== DATASET ======================
class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, groups=None):
        self.root_dir = root_dir
        self.transform = transform
        self.groups = groups or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.pairs = []

    @torch.no_grad()
    def initialize_pairs(self, model, device, build_chains=True, sim_threshold=0.97):
        model.eval()
        self.pairs = []
        desc = "Формируем train пары" if build_chains else "Формируем val пары"
        skipped_sim = 0

        for group in tqdm(self.groups, desc=desc):
            group_path = os.path.join(self.root_dir, group)
            png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
            best_txt = os.path.join(group_path, 'best.txt')

            if not os.path.exists(best_txt) or len(png_files) < 5:
                continue

            with open(best_txt, 'r') as f:
                best_file = f.read().strip()
            if best_file not in png_files:
                continue

            best_idx = png_files.index(best_file)
            other_indices = [i for i in range(len(png_files)) if i != best_idx]

            # Эмбеддинги для группы
            imgs = [val_transform(image=np.array(Image.open(os.path.join(group_path, png_files[idx])).convert('RGB')))['image']
                    for idx in [best_idx] + other_indices]

            batch = torch.stack(imgs).to(device)
            features = model.backbone(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            best_feat = features[0:1]
            others_feat = features[1:]
            sims = torch.mm(best_feat, others_feat.t()).squeeze(0)

            # Пары лидер vs остальные
            for i, sim in enumerate(sims.cpu().numpy()):
                idx_other = other_indices[i]
                if build_chains and sim > sim_threshold:
                    skipped_sim += 2
                    continue
                self.pairs.append((group, best_idx, idx_other, 1))
                if build_chains:
                    self.pairs.append((group, idx_other, best_idx, -1))

            # Цепочки проигравших
            if build_chains:
                sorted_sub_idx = torch.argsort(sims, descending=True).cpu().numpy()
                ranked_others = [other_indices[i] for i in sorted_sub_idx]

                for i in range(len(ranked_others) - 1):
                    idx_w = ranked_others[i]
                    idx_l = ranked_others[i + 1]
                    feat_w = features[1 + sorted_sub_idx[i]]
                    feat_l = features[1 + sorted_sub_idx[i + 1]]
                    pair_sim = torch.dot(feat_w, feat_l).item()

                    if pair_sim > sim_threshold:
                        skipped_sim += 2
                        continue

                    self.pairs.append((group, idx_w, idx_l, 1))
                    self.pairs.append((group, idx_l, idx_w, -1))

        if build_chains:
            random.shuffle(self.pairs)

        print(f"Создано пар: {len(self.pairs)}. Отсеяно слишком похожих: {skipped_sim}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        group, idx_a, idx_b, target = self.pairs[idx]
        group_path = os.path.join(self.root_dir, group)
        png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])

        img_a = np.array(Image.open(os.path.join(group_path, png_files[idx_a])).convert('RGB'))
        img_b = np.array(Image.open(os.path.join(group_path, png_files[idx_b])).convert('RGB'))

        if self.transform:
            img_a = self.transform(image=img_a)['image']
            img_b = self.transform(image=img_b)['image']

        return img_a, img_b, torch.tensor(target, dtype=torch.float32), group


# ====================== МОДЕЛЬ ======================
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

        self.rank_head = nn.Sequential(
            nn.Linear(4096, 1024),
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
        batch_size = x.size(0)
        num_images = x.size(1) if x.dim() == 5 else 1

        x = x.view(-1, 3, x.size(-2), x.size(-1))
        features = self.backbone(x)
        features = features.view(batch_size, num_images, -1)

        flat_features = features.view(batch_size * num_images, -1)
        flat_scores = self.rank_head(flat_features)
        scores = flat_scores.view(batch_size, num_images)

        return scores.squeeze(-1) if num_images == 1 else scores


# ====================== LOSS ======================
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


# ====================== ВСПОМОГАТЕЛЬНЫЕ ======================
def read_groups_from_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def my_collate_fn(batch):
    best_images = torch.stack([item[0] for item in batch])
    other_images = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    group_names = [item[3] for item in batch]
    return best_images, other_images, targets, group_names


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
                gid = group_names[i]
                if gid not in group_images_dict:
                    group_images_dict[gid] = []
                group_images_dict[gid].append(best_images[i:i+1].cpu())
                group_images_dict[gid].append(other_images[i:i+1].cpu())

        for group_id, images_list in group_images_dict.items():
            if len(images_list) < 8:
                continue

            unique_imgs = []
            seen = set()
            for img_tensor in images_list:
                img_hash = hash(img_tensor.numpy().tobytes())
                if img_hash not in seen:
                    seen.add(img_hash)
                    unique_imgs.append(img_tensor)

            if len(unique_imgs) != 5:
                continue

            group_tensor = torch.cat(unique_imgs[:5], dim=0).to(device)
            scores = model(group_tensor.unsqueeze(1)).squeeze(-1)

            true_best_idx = 0
            ranked_indices = torch.argsort(scores, descending=True)

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
                print(f"Ошибка NDCG для группы {group_id}: {e}")

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


# ====================== TRAIN ======================
def train():
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)
    val_groups = read_groups_from_txt(os.path.join(BASE_DIR, "val.txt"))

    print(f"Режим обучения: **{CFG['train_mode'].upper()}**\n")

    # Создаём модель
    model = EnhancedAnimeRanker().to(CFG['device'])

    # ==================== Создание датасетов ====================
    if CFG['train_mode'] == 'manual':
        groups = read_groups_from_txt(os.path.join(BASE_DIR, "train.txt"))
        train_dataset = AnimeGroupDataset(CFG['dataset_path'], train_transform, groups)
        print(f"Используется только ручной датасет: {len(groups)} групп")
        print("\n--- Формируем пары для ручного датасета ---")
        train_dataset.initialize_pairs(model, CFG['device'], build_chains=True)

    elif CFG['train_mode'] == 'synth':
        groups = [d for d in os.listdir(CFG['retrain_dir']) if os.path.isdir(os.path.join(CFG['retrain_dir'], d))]
        if not groups:
            raise FileNotFoundError("retrain папка пуста!")
        train_dataset = AnimeGroupDataset(CFG['retrain_dir'], train_transform, groups)
        print(f"Используется только синтетический датасет: {len(groups)} групп")
        print("\n--- Формируем пары для синтетического датасета ---")
        train_dataset.initialize_pairs(model, CFG['device'], build_chains=True)

    elif CFG['train_mode'] == 'both':
        manual_groups = read_groups_from_txt(os.path.join(BASE_DIR, "train.txt"))
        synth_groups = [d for d in os.listdir(CFG['retrain_dir']) if os.path.isdir(os.path.join(CFG['retrain_dir'], d))]

        manual_ds = AnimeGroupDataset(CFG['dataset_path'], train_transform, manual_groups)
        synth_ds = AnimeGroupDataset(CFG['retrain_dir'], train_transform, synth_groups)

        print(f"Найдено групп: ручных — {len(manual_groups)}, синтетических — {len(synth_groups)}")

        print("\n--- Формируем пары для ручного датасета ---")
        manual_ds.initialize_pairs(model, CFG['device'], build_chains=True)

        print("\n--- Формируем пары для синтетического датасета ---")
        synth_ds.initialize_pairs(model, CFG['device'], build_chains=True)

        train_dataset = ConcatDataset([manual_ds, synth_ds])
        print(f"\nОбъединено: {len(manual_ds)} + {len(synth_ds)} = {len(train_dataset)} пар")

    else:
        raise ValueError("train_mode должен быть 'manual', 'synth' или 'both'")

    # Валидационный датасет
    val_dataset = AnimeGroupDataset(CFG['dataset_path'], val_transform, val_groups)
    print("\n--- Формируем пары для валидации ---")
    val_dataset.initialize_pairs(model, CFG['device'], build_chains=False)
    print("----------------------------------------\n")

    # ==================== DataLoader ====================
    train_loader = DataLoader(
        train_dataset,
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
        val_dataset,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )

    # ==================== Обучение ====================
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG['backbone_lr']},
        {'params': model.rank_head.parameters(), 'lr': CFG['head_lr']}
    ], weight_decay=CFG['weight_decay'])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=CFG['min_lr'])
    criterion = SoftFocalPairwiseLoss()
    scaler = GradScaler('cuda')

    best_ndcg = 0.0
    best_top1 = 0.0
    epochs_without_improvement = 0

    print("Начало обучения...\n")

    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_head_grad_norm = 0.0
        epoch_backbone_grad_norm = 0.0

        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1}/{CFG["epochs"]}')

        for best_images, other_images, targets, _ in progress_bar:
            best_images = best_images.to(CFG['device'], non_blocking=True)
            other_images = other_images.to(CFG['device'], non_blocking=True)
            targets = targets.to(CFG['device'], non_blocking=True)

            scores_best = model(best_images.unsqueeze(1))
            scores_other = model(other_images.unsqueeze(1))

            loss = criterion(scores_best, scores_other, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            head_norm = torch.nn.utils.clip_grad_norm_(model.rank_head.parameters(), CFG['grad_clip'])
            backbone_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), CFG['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            epoch_head_grad_norm += head_norm.item()
            epoch_backbone_grad_norm += backbone_norm.item()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_head_norm = epoch_head_grad_norm / len(train_loader)
        avg_backbone_norm = epoch_backbone_grad_norm / len(train_loader)

        val_metrics = evaluate(model, val_loader, CFG['device'])

        improved = False
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            best_top1 = val_metrics['top1']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            improved = True
            print(f"→ НОВАЯ ЛУЧШАЯ МОДЕЛЬ сохранена! NDCG: {best_ndcg:.4f} | Top-1: {best_top1:.4f}")

        else:
            epochs_without_improvement += 1

        print(f"\nЭпоха {epoch+1}/{CFG['epochs']} завершена")
        print(f"Train Loss         : {avg_train_loss:.4f}")
        print(f"Val Loss           : {val_metrics['val_loss']:.4f}")
        print(f"Val NDCG           : {val_metrics['ndcg']:.4f} {'↑' if improved else ''}")
        print(f"Val Top-1          : {val_metrics['top1']:.4f}")
        print(f"Val Top-2          : {val_metrics['top2']:.4f}")
        print(f"Head Grad Norm     : {avg_head_norm:.4f}")
        print(f"Backbone Grad Norm : {avg_backbone_norm:.4f}")
        print("-" * 80)

        if epochs_without_improvement >= CFG['patience']:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

        scheduler.step()

    torch.save(model.state_dict(), 'final_model.pth')
    print(f"\nОбучение завершено! Лучший NDCG на валидации: {best_ndcg:.4f}")


if __name__ == '__main__':
    train()