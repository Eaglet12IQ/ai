import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from new_top import (
    CFG, AnimeGroupDataset, train_transform, val_transform,
    my_collate_fn, worker_init_fn, EnhancedAnimeRanker,
    PairwiseRankingLoss, evaluate
)

def load_splits():
    """Читает готовые train/val сплиты (фиксированные)."""
    splits_dir = Path(CFG["dataset_path"]) / "splits"
    with open("train.txt") as f:
        train_groups = f.read().splitlines()
    with open("val.txt") as f:
        val_groups = f.read().splitlines()
    return train_groups, val_groups

def retrain():
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)

    # Загружаем сплиты
    train_groups, val_groups = load_splits()

    # Датасеты
    train_ds = AnimeGroupDataset(CFG['dataset_path'], transform=train_transform, groups=train_groups)
    val_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=val_groups)
    retrain_ds = AnimeGroupDataset(CFG['retrain_dir'], transform=train_transform)

    # Лоадеры
    train_loader = DataLoader(
        ConcatDataset([train_ds, retrain_ds]),
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
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

    # Загружаем лучшую модель
    model = EnhancedAnimeRanker().to(CFG['device'])
    model.load_state_dict(torch.load("7186 4182 6364 best.pth", map_location=CFG['device']))
    print("✅ Загружена модель best_model.pth")

    # --- уменьшенные learning rates для retrain ---
    retrain_head_lr = CFG['head_lr'] * 0.02       # в 5 раз меньше
    retrain_backbone_lr = CFG['backbone_lr'] * 0.05  # в 2 раза меньше

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': retrain_backbone_lr},
        {'params': model.rank_head.parameters(), 'lr': retrain_head_lr}
    ], weight_decay=CFG['weight_decay'])

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-7)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=CFG['min_lr'])
    criterion = PairwiseRankingLoss()

    best_ndcg = 0
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1}/{CFG["epochs"]}')

        for best_images, other_images, targets, _ in progress_bar:
            best_images = best_images.to(CFG['device'], non_blocking=True)
            other_images = other_images.to(CFG['device'], non_blocking=True)
            targets = targets.to(CFG['device'], non_blocking=True)

            scores_best = model(best_images.unsqueeze(1)).squeeze(-1)
            scores_other = model(other_images.unsqueeze(1)).squeeze(-1)
            loss = criterion(scores_best, scores_other, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)

        # Валидация
        val_metrics = evaluate(model, val_loader, CFG['device'])
        print(f"\n📊 Эпоха {epoch+1} | Средний loss={avg_loss:.4f} | "
              f"Val NDCG={val_metrics['ndcg']:.4f}, Top-1={val_metrics['top1']:.4f}, Top-2={val_metrics['top2']:.4f}\n")

        # SWA
        if epoch >= CFG['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_bn(train_loader, swa_model, device=CFG['device'])
            swa_metrics = evaluate(swa_model, val_loader, CFG['device'])
            print(f"SWA Val NDCG: {swa_metrics['ndcg']:.4f}, Top-1: {swa_metrics['top1']:.4f}")

        scheduler.step()

        # Сохранение лучшей модели
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 Новая лучшая модель сохранена (NDCG={best_ndcg:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"⏹ Ранняя остановка на {epoch+1}-й эпохе")
            break

    model.load_state_dict(torch.load("best_model.pth", map_location=CFG['device']))

if __name__ == "__main__":
    retrain()
