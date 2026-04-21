import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import ndcg_score
from tqdm import tqdm

class Config:
    SEED = 49
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 6
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 0.01

    HEAD_LR = 1e-4
    BACKBONE_LR = 5e-6
    MIN_LR = 5e-7
    EPOCHS = 150
    PATIENCE = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET_PATH = "dataset"
    RETRAIN_DIR = os.path.join("dataset", "retrain")

    TRAIN_MODE = 'synth'

    NORMALIZE_MEAN = [0.54288839, 0.52424041, 0.52013308]
    NORMALIZE_STD = [0.32821858, 0.31147094, 0.30761928]


def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.85, 1.0), ratio=(0.75, 1.35)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.03, 0.07),
            hole_width_range=(0.03, 0.07),
            fill=tuple(int(x * 255) for x in Config.NORMALIZE_MEAN),
            p=0.4
        ),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        A.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=360),
        A.CenterCrop(360, 360),
        A.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ToTensorV2()
    ])

    return train_transform, val_transform


class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, groups: List[str] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.groups = groups or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.pairs: List[Tuple[str, int, int, int]] = []

    @torch.no_grad()
    def initialize_pairs(self, model: nn.Module, device: torch.device, 
                        build_chains: bool = True, sim_threshold: float = 0.97):
        model.eval()
        self.pairs.clear()
        skipped_sim = 0
        desc = "Формируем train пары" if build_chains else "Формируем val пары"

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

            imgs = [val_transform(image=np.array(Image.open(os.path.join(group_path, png_files[idx])).convert('RGB')))['image']
                    for idx in [best_idx] + other_indices]

            batch = torch.stack(imgs).to(device)
            features = model.backbone(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            best_feat = features[0:1]
            others_feat = features[1:]
            sims = torch.mm(best_feat, others_feat.t()).squeeze(0)

            for i, sim in enumerate(sims.cpu().numpy()):
                idx_other = other_indices[i]
                if build_chains and sim > sim_threshold:
                    skipped_sim += 2
                    continue
                self.pairs.append((group, best_idx, idx_other, 1))
                if build_chains:
                    self.pairs.append((group, idx_other, best_idx, -1))

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

    def __getitem__(self, idx: int):
        group, idx_a, idx_b, target = self.pairs[idx]
        group_path = os.path.join(self.root_dir, group)
        png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])

        img_a = np.array(Image.open(os.path.join(group_path, png_files[idx_a])).convert('RGB'))
        img_b = np.array(Image.open(os.path.join(group_path, png_files[idx_b])).convert('RGB'))

        if self.transform:
            img_a = self.transform(image=img_a)['image']
            img_b = self.transform(image=img_b)['image']

        return img_a, img_b, torch.tensor(target, dtype=torch.float32), group


class EnhancedAnimeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        from danbooru_resnet import resnet50 as danbooru_resnet50

        weights_path = "resnet50danbooru.pth"
        self.backbone = danbooru_resnet50(pretrained=False, top_n=6000)
        state_dict = torch.load(weights_path, map_location=Config.DEVICE, weights_only=True)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        num_images = x.size(1) if x.dim() == 5 else 1

        x = x.view(-1, 3, x.size(-2), x.size(-1))
        features = self.backbone(x)
        features = features.view(batch_size, num_images, -1)

        flat_features = features.view(batch_size * num_images, -1)
        flat_scores = self.rank_head(flat_features)
        scores = flat_scores.view(batch_size, num_images)

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


def read_groups_from_txt(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def collate_fn(batch):
    best_images = torch.stack([item[0] for item in batch])
    other_images = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    group_names = [item[3] for item in batch]
    return best_images, other_images, targets, group_names


def worker_init_fn(worker_id):
    np.random.seed(Config.SEED + worker_id)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    criterion = SoftFocalPairwiseLoss()

    top1_correct = top2_correct = 0
    ndcg_scores = []
    val_loss_sum = val_loss_count = processed_groups = 0
    group_images_dict: Dict[str, List[torch.Tensor]] = {}

    for best_images, other_images, targets, group_names in tqdm(dataloader, desc="Evaluation", leave=False):
        best_images = best_images.to(device, non_blocking=True)
        other_images = other_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        scores_best = model(best_images.unsqueeze(1)).squeeze(-1)
        scores_other = model(other_images.unsqueeze(1)).squeeze(-1)

        batch_loss = criterion(scores_best, scores_other, targets)
        val_loss_sum += batch_loss.item() * best_images.size(0)
        val_loss_count += best_images.size(0)

        for i, gid in enumerate(group_names):
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
            ndcg = ndcg_score(true_relevance, scores.unsqueeze(0).cpu().numpy(), k=5)
            ndcg_scores.append(ndcg)
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


def train():
    set_seed()
    print(f"Using device: {Config.DEVICE} | Mode: {Config.TRAIN_MODE.upper()}\n")

    Path(Config.RETRAIN_DIR).mkdir(parents=True, exist_ok=True)

    global val_transform
    train_transform, val_transform = get_transforms()

    model = EnhancedAnimeRanker().to(Config.DEVICE)

    if Config.TRAIN_MODE == 'manual':
        groups = read_groups_from_txt("train.txt")
        train_dataset = AnimeGroupDataset(Config.DATASET_PATH, train_transform, groups)
        train_dataset.initialize_pairs(model, Config.DEVICE, build_chains=True)

    elif Config.TRAIN_MODE == 'synth':
        groups = [d for d in os.listdir(Config.RETRAIN_DIR) if os.path.isdir(os.path.join(Config.RETRAIN_DIR, d))]
        if not groups:
            raise FileNotFoundError("retrain папка пуста!")
        train_dataset = AnimeGroupDataset(Config.RETRAIN_DIR, train_transform, groups)
        train_dataset.initialize_pairs(model, Config.DEVICE, build_chains=True)

    elif Config.TRAIN_MODE == 'both':
        manual_groups = read_groups_from_txt("train.txt")
        synth_groups = [d for d in os.listdir(Config.RETRAIN_DIR) if os.path.isdir(os.path.join(Config.RETRAIN_DIR, d))]

        manual_ds = AnimeGroupDataset(Config.DATASET_PATH, train_transform, manual_groups)
        synth_ds = AnimeGroupDataset(Config.RETRAIN_DIR, train_transform, synth_groups)

        manual_ds.initialize_pairs(model, Config.DEVICE, build_chains=True)
        synth_ds.initialize_pairs(model, Config.DEVICE, build_chains=True)

        train_dataset = ConcatDataset([manual_ds, synth_ds])
        print(f"Объединено: {len(manual_ds)} + {len(synth_ds)} = {len(train_dataset)} пар")
    else:
        raise ValueError("train_mode должен быть 'manual', 'synth' или 'both'")

    val_groups = read_groups_from_txt("val.txt")
    val_dataset = AnimeGroupDataset(Config.DATASET_PATH, val_transform, val_groups)
    val_dataset.initialize_pairs(model, Config.DEVICE, build_chains=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': Config.BACKBONE_LR},
        {'params': model.rank_head.parameters(), 'lr': Config.HEAD_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=Config.MIN_LR)
    criterion = SoftFocalPairwiseLoss()
    scaler = GradScaler('cuda')

    best_ndcg = 0.0
    best_top1 = 0.0
    epochs_without_improvement = 0

    print("Начало обучения...\n")

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_head_grad_norm = 0.0
        epoch_backbone_grad_norm = 0.0

        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1}/{Config.EPOCHS}')

        for best_images, other_images, targets, _ in progress_bar:
            best_images = best_images.to(Config.DEVICE, non_blocking=True)
            other_images = other_images.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)

            scores_best = model(best_images.unsqueeze(1))
            scores_other = model(other_images.unsqueeze(1))
            loss = criterion(scores_best, scores_other, targets)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            head_norm = torch.nn.utils.clip_grad_norm_(model.rank_head.parameters(), Config.GRAD_CLIP)
            backbone_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), Config.GRAD_CLIP)

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

        val_metrics = evaluate(model, val_loader, Config.DEVICE)

        improved = val_metrics['ndcg'] > best_ndcg
        if improved:
            best_ndcg = val_metrics['ndcg']
            best_top1 = val_metrics['top1']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"→ НОВАЯ ЛУЧШАЯ МОДЕЛЬ сохранена! NDCG: {best_ndcg:.4f} | Top-1: {best_top1:.4f}")
        else:
            epochs_without_improvement += 1

        print(f"\nЭпоха {epoch+1}/{Config.EPOCHS} завершена")
        print(f"Train Loss         : {avg_train_loss:.4f}")
        print(f"Val Loss           : {val_metrics['val_loss']:.4f}")
        print(f"Val NDCG           : {val_metrics['ndcg']:.4f} {'↑' if improved else ''}")
        print(f"Val Top-1          : {val_metrics['top1']:.4f}")
        print(f"Val Top-2          : {val_metrics['top2']:.4f}")
        print(f"Head Grad Norm     : {avg_head_norm:.4f}")
        print(f"Backbone Grad Norm : {avg_backbone_norm:.4f}")
        print("-" * 90)

        if epochs_without_improvement >= Config.PATIENCE:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

        scheduler.step()

    torch.save(model.state_dict(), 'final_model.pth')
    print(f"\nОбучение завершено! Лучший NDCG: {best_ndcg:.4f}")


if __name__ == '__main__':
    train()