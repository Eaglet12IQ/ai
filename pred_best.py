import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

plt.rcParams['font.family'] = 'Segoe UI Emoji'  # Или укажите другой подходящий шрифт

# Путь к папке, где находится скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def my_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])
    return images, labels

# Конфигурация
CFG = {
    'model_name': 'swin_small_patch4_window7_224',  # Уменьшили модель
    'img_size': 224,  # Уменьшили размер изображения
    'batch_size': 16,
    'num_workers': 12,
    'warmup_epochs': 5,
    'grad_clip': 1.0,  # Ослабить клиппинг
    'weight_decay': 0.2,  # Уменьшить регуляризацию
    'max_lr': 1e-5,  # Увеличить в 3 раза
    'head_lr': 2e-4,
    'min_lr': 1e-6,
    'epochs': 50,  # Увеличили количество эпох
    'swa_start': 60,  # С какой эпохи начинаем SWA (например, половина обучения)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'retrain_dir': os.path.join(BASE_DIR, "dataset", "retrain"),
    'test_size': 0.1,
    'val_size': 0.1,
    'dataset_path': os.path.join(BASE_DIR, "dataset"),
}

# Улучшенные аугментации
train_transform = A.Compose([
    A.Resize(224, 224),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 5),
        hole_height_range=(0.05, 0.1),
        hole_width_range=(0.05, 0.1),
        fill=(0.485*255, 0.456*255, 0.406*255),
        p=0.5
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.LongestMaxSize(CFG['img_size']),
    A.PadIfNeeded(CFG['img_size'], CFG['img_size'], border_mode=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class AnimeGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, groups=None):
        self.root_dir = root_dir
        self.transform = transform

        # Если groups не переданы, получаем все папки из root_dir
        all_groups = groups or [d for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))]
        
        # Фильтруем группы: оставляем только те, где есть 5 изображений и корректный файл best.txt
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
        self.backbone = create_model(
            CFG['model_name'],
            pretrained=True,
            num_classes=192,  # Уменьшили размер фичей
            drop_path_rate=0.05  # Меньше дропаут для Tiny
        )
        self.rank_head = nn.Sequential(
            nn.Linear(192, 96),  # Соответственно изменили размерности
            nn.GELU(),
            nn.Dropout(0.4),     # Добавили дропаут в голову
            nn.Linear(96, 1)
        )
        
    def forward(self, x_group):
        batch_size = x_group.size(0)
        features = self.backbone(x_group.flatten(0, 1))
        features = features.view(batch_size, 5, -1)
        return self.rank_head(features).squeeze(-1)

class RankNetLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.margin = margin
    
    def forward(self, scores, targets):
        batch_size = scores.size(0)
        loss = 0
        for i in range(batch_size):
            pos_score = scores[i, targets[i]]
            for j in range(5):
                if j == targets[i]:
                    continue
                neg_score = scores[i, j]
                diff = pos_score - neg_score - self.margin
                loss += -torch.log(self.sigmoid(diff))
        return loss / (batch_size * 4)

def train():
    # Инициализация директории для дообучения
    Path(CFG['retrain_dir']).mkdir(parents=True, exist_ok=True)
    
    # Загрузка и разделение данных
    all_groups = [d for d in os.listdir(CFG['dataset_path']) 
                  if os.path.isdir(os.path.join(CFG['dataset_path'], d))]
    
    train_groups, test_groups = train_test_split(
        all_groups, 
        test_size=CFG['test_size'],
        random_state=42
    )
    train_groups, val_groups = train_test_split(
        train_groups, 
        test_size=CFG['val_size'],
        random_state=42
    )
    
    # Создание датасетов
    train_ds = AnimeGroupDataset(CFG['dataset_path'], transform=train_transform, groups=train_groups)
    val_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=val_groups)
    test_ds = AnimeGroupDataset(CFG['dataset_path'], transform=val_transform, groups=test_groups)
    retrain_ds = AnimeGroupDataset(CFG['retrain_dir'], transform=train_transform)
    
    # DataLoader-ы
    train_loader = DataLoader(
        ConcatDataset([train_ds, retrain_ds]),
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(  # Добавляем test_loader здесь
        test_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True
    )
    
    # Инициализация модели, оптимизатора и scheduler-ов
    model = EnhancedAnimeRanker().to(CFG['device'])
    optimizer = torch.optim.AdamW(
        [
            {'params': model.backbone.parameters(), 'lr': CFG['max_lr']},
            {'params': model.rank_head.parameters(), 'lr': CFG['head_lr']}
        ],
        weight_decay=CFG['weight_decay']
    )
    
    # Создаём SWA-модель (она будет обновляться начиная с swa_start)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-6)
    
    # OneCycleLR для первой фазы обучения
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        eta_min=CFG['min_lr']
    )
    
    criterion = RankNetLoss()
    
    best_ndcg = 0
    patience = 10
    epochs_without_improvement = 0
    gradient_accumulation_steps = 2  # Эффективный batch_size = 16 * 2 = 32
    history = {
        'train_loss': [],
        'val_ndcg': [],
        'val_top1': []
    }
    
    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CFG["epochs"]}')
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(CFG['device'], non_blocking=True)
            labels = labels.to(CFG['device'], non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)  # Реальный лосс батча (~0.7)
            loss_for_backward = loss / gradient_accumulation_steps  # Лосс для градиентов (~0.35)
            loss_for_backward.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()  # Суммируем реальный лосс батча
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # Показываем реальный лосс батча
        
        if (i + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        scheduler.step()
        val_metrics = evaluate(model, val_loader, CFG['device'])
        
        history['train_loss'].append(epoch_loss / len(train_loader))  # Средний лосс батча
        history['val_ndcg'].append(val_metrics['ndcg'])
        history['val_top1'].append(val_metrics['top1'])
        
        if val_metrics['ndcg'] > best_ndcg:
            best_ndcg = val_metrics['ndcg']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_NDCG_model.pth')
            print(f"New best model saved with NDCG: {best_ndcg:.4f}")
            
            print("\nEvaluating best model on test set...")
            test_model = EnhancedAnimeRanker().to(CFG['device'])
            test_model.load_state_dict(torch.load('best_NDCG_model.pth'))
            test_metrics = evaluate(test_model, test_loader, CFG['device'])
            print(f"Test NDCG: {test_metrics['ndcg']:.4f}")
            print(f"Test Top-1 Accuracy: {test_metrics['top1']:.4f}\n")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val NDCG: {val_metrics['ndcg']:.4f}")
        print(f"Val Top-1 Accuracy: {val_metrics['top1']:.4f}")
        print("-" * 50)

    # После окончания обучения
    if CFG['swa_start'] < CFG['epochs']:
        update_bn(train_loader, swa_model, device=CFG['device'])
        final_model = swa_model.module  # Берем внутреннюю модель
    else:
        final_model = model

    # Сохраняем финальную модель без SWA обертки
    torch.save(final_model.state_dict(), 'final_model.pth')

    # Тестирование SWA-модели с правильной загрузкой
    print("\nFinal Evaluation of Final Model on Test Set:")
    swa_model = EnhancedAnimeRanker().to(CFG['device'])
    swa_model.load_state_dict(torch.load('final_model.pth'))  # Без SWA параметров
    swa_model.eval()
    test_metrics_swa = evaluate(swa_model, test_loader, CFG['device'])
    print(f"Test NDCG: {test_metrics_swa['ndcg']:.4f}")
    print(f"Test Top-1 Accuracy: {test_metrics_swa['top1']:.4f}")

    # Тестирование лучшей модели
    print("\nFinal Evaluation of Best NDCG Model on Test Set:")
    best_ndcg_model = EnhancedAnimeRanker().to(CFG['device'])
    best_ndcg_model.load_state_dict(torch.load('best_NDCG_model.pth'))
    best_ndcg_model.eval()
    test_metrics_best = evaluate(best_ndcg_model, test_loader, CFG['device'])
    print(f"Test NDCG: {test_metrics_best['ndcg']:.4f}")
    print(f"Test Top-1 Accuracy: {test_metrics_best['top1']:.4f}")
    
    # Визуализация кривых обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_ndcg'], label='Validation NDCG')
    plt.plot(history['val_top1'], label='Validation Top-1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Визуализация примеров предсказаний
    visualize_predictions(final_model, test_ds, num_examples=5)

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

def visualize_predictions(model, dataset, num_examples=3):
    model.eval()
    indices = np.random.choice(len(dataset), num_examples)
    
    plt.figure(figsize=(15, 5 * num_examples))
    for plot_idx, data_idx in enumerate(indices):
        images, true_label = dataset[data_idx]
        with torch.no_grad():
            scores = model(images.unsqueeze(0).to(CFG['device'])).cpu().numpy()[0]
        
        for i in range(5):
            plt.subplot(num_examples, 5, plot_idx * 5 + i + 1)
            img = images[i].permute(1, 2, 0).numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f"Score: {scores[i]:.2f}\n{'✅' if i == true_label else '❌'}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def predict_with_hitl(model_path, group_folder, temp=1):
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
    dynamic_threshold = 0.3 + 0.5 * (1 - sorted_probs[0])
    print(sorted_probs)
    print(confidence_ratio)
    print(1 + dynamic_threshold)
    
    if confidence_ratio < (1 + dynamic_threshold):
        print(f"\n⚠️ Требуется ручная проверка группы: {group_folder}")
        print("Изображения:")
        for i, img_file in enumerate(png_files):
            print(f"{i+1}. {img_file}")
        while True:
            try:
                choice = int(input("Введите номер лучшего изображения: ")) - 1
                if 0 <= choice < len(png_files):
                    best_file = png_files[choice]
                    save_for_retraining(group_folder, best_file)
                    return best_file
                else:
                    print("Некорректный ввод. Попробуйте снова.")
            except ValueError:
                print("Пожалуйста, введите число.")
    else:
        best_idx = torch.argmax(scores).item()
        return png_files[best_idx]

def save_for_retraining(group_folder, best_file):
    group_id = os.path.basename(group_folder)
    dest_folder = os.path.join(CFG['retrain_dir'], group_id)
    if os.path.exists(dest_folder):
        return
    os.makedirs(dest_folder)
    for img_file in os.listdir(group_folder):
        if img_file.endswith('.png'):
            src = os.path.join(group_folder, img_file)
            dst = os.path.join(dest_folder, img_file)
            shutil.copy2(src, dst)
    with open(os.path.join(dest_folder, 'best.txt'), 'w') as f:
        f.write(best_file)

if __name__ == '__main__':
    train()
    # Пример использования:
    # test_group = os.path.join(BASE_DIR, "input")
    # best_image = predict_with_hitl('best_NDCG_model.pth', test_group)
    # print(f"\n🎯 Лучшее изображение: {best_image}")