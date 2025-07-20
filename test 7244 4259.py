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
from sklearn.model_selection import KFold, train_test_split
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

def my_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])
    return images, labels

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

CFG = {
    'img_size': 360,
    'batch_size': 32,
    'num_workers': 6,
    'grad_clip': 0.5,  # Уменьшен для большей стабильности
    'weight_decay': 0.05,
    'head_lr': 1e-4,   # Уменьшен для головы
    'backbone_lr': 1e-5,  # Уменьшен для backbone
    'min_lr': 5e-7,
    'epochs': 150,
    'swa_start': 80,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
    'test_size': 0.1,
    'n_folds': 5,
    'dataset_path': os.path.join(BASE_DIR, "dataset")
}

# Обновленные аугментации
train_transform = A.Compose([
    A.Resize(360, 360),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.07),
        hole_width_range=(0.03, 0.07),
        fill=(0.7137*255, 0.6628*255, 0.6519*255),
        p=0.3
    ),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ToTensorV2()
], seed=seed)

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ToTensorV2()
], seed=seed)

class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, groups=None):
        self.root_dir = root_dir
        self.transform = transform
        all_groups = groups or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.groups = []
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
                else:
                    print(f"Warning: Invalid best.txt in {group_path}, file {best_file} not found")

    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        group_path = os.path.join(self.root_dir, self.groups[idx])
        images = []
        png_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
        for img_file in png_files:
            img_path = os.path.join(group_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                print(f"Warning: NaN or Inf in image {img_path}")
                img = np.zeros_like(img)  # Заменяем на нулевой массив
            if self.transform:
                img = self.transform(image=img)['image']
            images.append(img)
        with open(os.path.join(group_path, 'best.txt'), 'r') as f:
            best_file = f.read().strip()
        best_idx = png_files.index(best_file)
        return torch.stack(images), best_idx

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

class CombinedLoss(nn.Module):
    def __init__(self, target_prob=0.8, margin=1.0, top1_weight=0.7, pairwise_weight=0.3, epsilon=1e-6):
        super().__init__()
        self.target_prob = target_prob
        self.margin = margin
        self.top1_weight = top1_weight
        self.pairwise_weight = pairwise_weight
        self.epsilon = epsilon
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, scores, targets):
        # Top1ProbabilityLoss
        probs = torch.softmax(scores, dim=1)
        top1_probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1)).squeeze(1)
        top1_loss = -torch.mean(
            self.target_prob * torch.log(top1_probs + self.epsilon) +
            (1.0 - self.target_prob) * torch.log(1.0 - top1_probs + self.epsilon)
        )
        
        # PairwiseLoss
        pairwise_loss = 0.0
        batch_size = scores.size(0)
        for i in range(batch_size):
            best_idx = targets[i]
            best_score = scores[i, best_idx]
            for j in range(5):
                if j != best_idx:
                    diff = best_score - scores[i, j] - self.margin
                    diff = torch.clamp(diff, min=-10.0, max=10.0)
                    pairwise_loss += -torch.log(self.sigmoid(diff) + self.epsilon)
        pairwise_loss = pairwise_loss / (batch_size * 4)
        
        # Combined loss
        loss = self.top1_weight * top1_loss + self.pairwise_weight * pairwise_loss
        
        return loss

def load_and_evaluate(model_path, model_class, loader, device):
    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    model.load_state_dict(state_dict)
    model.eval()
    metrics = evaluate(model, loader, device)
    return metrics

def evaluate(model, dataloader, device):
    model.eval()
    top1_correct = 0
    top2_correct = 0
    ndcg_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            scores = model(images)
            
            ranked_indices = torch.argsort(scores, dim=1, descending=True)
            top1_correct += (ranked_indices[:, 0] == labels).sum().item()
            top2_correct += sum([1 for i in range(len(labels)) if labels[i] in ranked_indices[i, :2]])
            
            batch_true = np.zeros(scores.shape)
            for i, label in enumerate(labels.cpu().numpy()):
                batch_true[i, label] = 1
                
            try:
                batch_ndcg = ndcg_score(batch_true, scores.cpu().numpy(), k=5)
                ndcg_scores.append(batch_ndcg)
            except Exception as e:
                print(f"Error in NDCG: {e}")
                continue
                
    mean_ndcg = np.nanmean(ndcg_scores) if ndcg_scores else 0.0
    return {
        'top1': top1_correct / len(dataloader.dataset),
        'top2': top2_correct / len(dataloader.dataset),
        'ndcg': mean_ndcg
    }

def visualize_predictions(model, dataset, num_examples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    plt.figure(figsize=(15, 5 * num_examples))
    for plot_idx, data_idx in enumerate(indices):
        images, true_label = dataset[data_idx]
        with torch.no_grad():
            scores = model(images.unsqueeze(0).to(CFG['device'])).cpu().numpy()[0]
        
        for i in range(5):
            plt.subplot(num_examples, 5, plot_idx * 5 + i + 1)
            img = images[i].permute(1, 2, 0).numpy()
            img = img * np.array([0.2970, 0.3017, 0.2979]) + np.array([0.7137, 0.6628, 0.6519])
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f"Score: {scores[i]:.2f}\n{'✅' if i == true_label else '❌'}")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def train():
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)
    
    all_groups = [d for d in os.listdir(CFG['dataset_path']) 
                  if os.path.isdir(os.path.join(CFG['dataset_path'], d))]
    
    train_val_groups, test_groups = train_test_split(
        all_groups, 
        test_size=CFG['test_size'],
        random_state=seed
    )
    
    test_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=test_groups)
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    
    kfold = KFold(n_splits=CFG['n_folds'], shuffle=True, random_state=seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_groups)):
        print(f"\nStarting Fold {fold+1}/{CFG['n_folds']}")
        
        train_groups = [train_val_groups[i] for i in train_idx]
        val_groups = [train_val_groups[i] for i in val_idx]
        
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
        
        model = EnhancedAnimeRanker().to(CFG['device'])
        last_block = model.backbone[0][7]
        
        # Проверка весов на NaN/Inf
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf in weights of {name}")
        
        torch.save(model.state_dict(), f'initial_model_fold_{fold+1}.pth')
        
        optimizer = torch.optim.AdamW([
            {'params': last_block.parameters(), 'lr': CFG['backbone_lr']},
            {'params': model.rank_head.parameters(), 'lr': CFG['head_lr']}
        ], weight_decay=CFG['weight_decay'])
        
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=5e-7)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=CFG['min_lr'])
        criterion = CombinedLoss()
        scaler = GradScaler('cuda')
        
        best_ndcg = 0
        best_top1 = 0
        best_swa_ndcg = 0
        patience = 20
        epochs_without_improvement = 0
        history = {
            'train_loss': [],
            'val_ndcg': [],
            'val_top1': [],
            'swa_val_ndcg': [],
            'swa_val_top1': []
        }
        
        for epoch in range(CFG['epochs']):
            model.train()
            epoch_loss = 0
            epoch_head_grad_norm = 0
            epoch_backbone_grad_norm = 0
            progress_bar = tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{CFG["epochs"]}')
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(CFG['device'], non_blocking=True)
                labels = labels.to(CFG['device'], non_blocking=True)
                batch_groups = train_ds.groups[batch_idx * CFG['batch_size']:(batch_idx + 1) * CFG['batch_size']]
                
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"Warning: NaN or Inf detected in input images at batch {progress_bar.n}")
                    continue
                
                outputs = model(images)  # Без autocast для стабильности
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN or Inf detected in model outputs at batch {progress_bar.n}")
                    continue
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or Inf detected in loss at batch {progress_bar.n}")
                    optimizer.zero_grad(set_to_none=True)  # Clear gradients to avoid propagating NaN/Inf
                    continue
                
                loss.backward()
                head_grad_norm = torch.nn.utils.clip_grad_norm_(model.rank_head.parameters(), CFG['grad_clip'])
                backbone_grad_norm = torch.nn.utils.clip_grad_norm_(last_block.parameters(), CFG['grad_clip'])
                
                if torch.isnan(head_grad_norm) or torch.isinf(head_grad_norm) or \
                   torch.isnan(backbone_grad_norm) or torch.isinf(backbone_grad_norm):
                    print(f"Warning: NaN/Inf gradient in epoch {epoch+1}, batch {progress_bar.n}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                if head_grad_norm > 10.0 or backbone_grad_norm > 10.0:
                    print(f"Warning: Large gradients (head: {head_grad_norm:.2f}, backbone: {backbone_grad_norm:.2f})")
                    print(batch_groups)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                epoch_loss += loss.item()
                epoch_head_grad_norm += head_grad_norm.item()
                epoch_backbone_grad_norm += backbone_grad_norm.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'       
                })
            
            avg_head_grad_norm = epoch_head_grad_norm / len(train_loader)
            avg_backbone_grad_norm = epoch_backbone_grad_norm / len(train_loader)
            
            val_metrics = evaluate(model, val_loader, CFG['device'])
            
            history['train_loss'].append(epoch_loss / len(train_loader))
            history['val_ndcg'].append(val_metrics['ndcg'])
            history['val_top1'].append(val_metrics['top1'])
            
            if epoch >= CFG['swa_start']:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                update_bn(train_loader, swa_model, device=CFG['device'])
                swa_metrics = evaluate(swa_model, val_loader, CFG['device'])
                history['swa_val_ndcg'].append(swa_metrics['ndcg'])
                history['swa_val_top1'].append(swa_metrics['top1'])
                print(f"Fold {fold+1} SWA Val NDCG: {swa_metrics['ndcg']:.4f}, SWA Val Top-1: {swa_metrics['top1']:.4f}")
                if swa_metrics['ndcg'] > best_swa_ndcg:
                    best_swa_ndcg = swa_metrics['ndcg']
                    torch.save(swa_model.module.state_dict(), f'best_swa_model_fold_{fold+1}.pth')
                    print(f"Fold {fold+1} New best SWA model saved with NDCG: {best_swa_ndcg:.4f}")
            else:
                history['swa_val_ndcg'].append(0.0)
                history['swa_val_top1'].append(0.0)
            
            scheduler.step(val_metrics['ndcg'])
            
            if val_metrics['ndcg'] > best_ndcg:
                best_ndcg = val_metrics['ndcg']
                best_top1 = val_metrics['top1']
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'best_NDCG_model_fold_{fold+1}.pth')
                print(f"Fold {fold+1} New best model saved with NDCG: {best_ndcg:.4f}, Top-1: {best_top1:.4f}")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Fold {fold+1} Early stopping at epoch {epoch+1}")
                break
            
            print(f"\nFold {fold+1} Epoch {epoch+1} Summary:")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val NDCG: {val_metrics['ndcg']:.4f}")
            print(f"Val Top-1 Accuracy: {val_metrics['top1']:.4f}")
            print(f"Avg Head Grad Norm: {avg_head_grad_norm:.4f}")
            print(f"Avg Backbone Grad Norm: {avg_backbone_grad_norm:.4f}")
            print("-" * 50)
        
        fold_results.append({
            'fold': fold + 1,
            'val_ndcg': best_ndcg,
            'val_top1': best_top1
        })
        
        torch.save(model.state_dict(), f'final_model_fold_{fold+1}.pth')
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title(f'Fold {fold+1} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_ndcg'], label='Validation NDCG')
        plt.plot(history['val_top1'], label='Validation Top-1')
        plt.plot(history['swa_val_ndcg'], label='SWA Validation NDCG', linestyle='--')
        plt.plot(history['swa_val_top1'], label='SWA Validation Top-1', linestyle='--')
        plt.title(f'Fold {fold+1} Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_fold_{fold+1}.png')
        plt.close()
    
    print("\nCross-Validation Summary:")
    avg_val_ndcg = np.mean([res['val_ndcg'] for res in fold_results])
    avg_val_top1 = np.mean([res['val_top1'] for res in fold_results])
    val_ndcg_std = np.std([res['val_ndcg'] for res in fold_results])
    val_top1_std = np.std([res['val_top1'] for res in fold_results])
    print(f"Average Validation NDCG: {avg_val_ndcg:.4f} (Std: {val_ndcg_std:.4f})")
    print(f"Average Validation Top-1 Accuracy: {avg_val_top1:.4f} (Std: {val_top1_std:.4f})")
    
    print("\nTraining Final Model")
    
    final_train_groups, final_val_groups = train_test_split(
        train_val_groups, test_size=0.1, random_state=seed
    )
    
    final_train_ds = AnimeGroupDataset(CFG['dataset_path'], transform=train_transform, groups=final_train_groups)
    final_val_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=final_val_groups)
    retrain_ds = AnimeGroupDataset(CFG['retrain_dir'], transform=train_transform)
    
    final_train_loader = DataLoader(
        ConcatDataset([final_train_ds, retrain_ds]),
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    
    final_val_loader = DataLoader(
        final_val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    
    final_model = EnhancedAnimeRanker().to(CFG['device'])
    last_block = final_model.backbone[0][7]
    
    optimizer = torch.optim.AdamW([
        {'params': last_block.parameters(), 'lr': CFG['backbone_lr']},
        {'params': final_model.rank_head.parameters(), 'lr': CFG['head_lr']}
    ], weight_decay=CFG['weight_decay'])
    
    swa_model = AveragedModel(final_model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-7)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=CFG['min_lr'])
    criterion = CombinedLoss()
    
    best_ndcg = 0
    best_top1 = 0
    best_swa_ndcg = 0
    patience = 20
    epochs_without_improvement = 0
    final_history = {
        'train_loss': [],
        'val_ndcg': [],
        'val_top1': [],
        'swa_val_ndcg': [],
        'swa_val_top1': []
    }
    
    for epoch in range(CFG['epochs']):
        final_model.train()
        epoch_loss = 0
        epoch_head_grad_norm = 0
        epoch_backbone_grad_norm = 0
        progress_bar = tqdm(final_train_loader, desc=f'Final Model Epoch {epoch+1}/{CFG["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(CFG['device'], non_blocking=True)
            labels = labels.to(CFG['device'], non_blocking=True)
            batch_groups = train_ds.groups[batch_idx * CFG['batch_size']:(batch_idx + 1) * CFG['batch_size']]
            
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"Warning: NaN or Inf detected in input images at batch {progress_bar.n}")
                continue
                
            outputs = final_model(images)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or Inf detected in model outputs at batch {progress_bar.n}")
                continue
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf detected in loss at batch {progress_bar.n}")
                optimizer.zero_grad(set_to_none=True)  # Clear gradients to avoid propagating NaN/Inf
                continue
            
            loss.backward()
            head_grad_norm = torch.nn.utils.clip_grad_norm_(final_model.rank_head.parameters(), CFG['grad_clip'])
            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(last_block.parameters(), CFG['grad_clip'])
            
            if torch.isnan(head_grad_norm) or torch.isinf(head_grad_norm) or \
               torch.isnan(backbone_grad_norm) or torch.isinf(backbone_grad_norm):
                print(f"Warning: NaN/Inf gradient in epoch {epoch+1}, batch {progress_bar.n}")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            if head_grad_norm > 10.0 or backbone_grad_norm > 10.0:
                print(f"Warning: Large gradients (head: {head_grad_norm:.2f}, backbone: {backbone_grad_norm:.2f})")
                print(batch_groups)
                optimizer.zero_grad(set_to_none=True)
                continue
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            epoch_head_grad_norm += head_grad_norm.item()
            epoch_backbone_grad_norm += backbone_grad_norm.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        avg_head_grad_norm = epoch_head_grad_norm / len(final_train_loader)
        avg_backbone_grad_norm = epoch_backbone_grad_norm / len(final_train_loader)
        
        val_metrics = evaluate(final_model, final_val_loader, CFG['device'])
        
        final_history['train_loss'].append(epoch_loss / len(final_train_loader))
        final_history['val_ndcg'].append(val_metrics['ndcg'])
        final_history['val_top1'].append(val_metrics['top1'])
        
        if epoch >= CFG['swa_start']:
            swa_model.update_parameters(final_model)
            swa_scheduler.step()
            update_bn(final_train_loader, swa_model, device=CFG['device'])
            swa_metrics = evaluate(swa_model, final_val_loader, CFG['device'])
            final_history['swa_val_ndcg'].append(swa_metrics['ndcg'])
            final_history['swa_val_top1'].append(swa_metrics['top1'])
            print(f"Final Model SWA Val NDCG: {swa_metrics['ndcg']:.4f}, SWA Val Top-1: {swa_metrics['top1']:.4f}")
            if swa_metrics['ndcg'] > best_swa_ndcg:
                best_swa_ndcg = swa_metrics['ndcg']
                torch.save(swa_model.module.state_dict(), 'best_swa_final_model.pth')
                print(f"New best SWA final model saved with NDCG: {best_swa_ndcg:.4f}")
        else:
            final_history['swa_val_ndcg'].append(0.0)
            final_history['swa_val_top1'].append(0.0)
        
        scheduler.step(val_metrics['ndcg'])
        
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            best_top1 = val_metrics['top1']
            epochs_without_improvement = 0
            torch.save(final_model.state_dict(), 'best_final_model.pth')
            print(f"New best final model saved with NDCG: {best_ndcg:.4f}, Top-1: {best_top1:.4f}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Final Model Early stopping at epoch {epoch+1}")
            break
        
        print(f"\nFinal Model Epoch {epoch+1} Summary:")
        print(f"Train Loss: {final_history['train_loss'][-1]:.4f}")
        print(f"Val NDCG: {val_metrics['ndcg']:.4f}")
        print(f"Val Top-1 Accuracy: {val_metrics['top1']:.4f}")
        print(f"Avg Head Grad Norm: {avg_head_grad_norm:.4f}")
        print(f"Avg Backbone Grad Norm: {avg_backbone_grad_norm:.4f}")
        print("-" * 50)
    
    print("\nFinal Evaluation on Test Set:")
    test_metrics_final = load_and_evaluate('best_final_model.pth', EnhancedAnimeRanker, test_loader, CFG['device'])
    print(f"Final Model Test NDCG: {test_metrics_final['ndcg']:.4f}")
    print(f"Final Model Test Top-1 Accuracy: {test_metrics_final['top1']:.4f}")
    
    if os.path.exists('best_swa_final_model.pth'):
        test_metrics_swa = load_and_evaluate('best_swa_final_model.pth', EnhancedAnimeRanker, test_loader, CFG['device'])
        print(f"Final SWA Model Test NDCG: {test_metrics_swa['ndcg']:.4f}")
        print(f"Final SWA Model Test Top-1 Accuracy: {test_metrics_swa['top1']:.4f}")
    
    final_model = EnhancedAnimeRanker().to(CFG['device'])
    final_model.load_state_dict(torch.load('best_final_model.pth', map_location=CFG['device']))
    visualize_predictions(final_model, test_ds)

if __name__ == '__main__':
    train()