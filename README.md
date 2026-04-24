# Anime Image Ranking & Generation System

Система для генерации и автоматического отбора лучших аниме-изображений с использованием нейросетей.

## Содержание

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Структура проекта](#структура-проекта)
3. [Требования](#требования)
4. [Быстрый старт](#быстрый-старт)
5. [Формат данных](#формат-данных)
6. [Архитектура модели](#архитектура-модели)
7. [Варианты обучения](#варианты-обучения)
8. [Loss функции](#loss-функции)
9. [Метрики](#метрики)
10. [Конфигурация](#конфигурация)
11. [ComfyUI Workflows](#comfyui-workflows)
12. [База данных](#база-данных)
13. [Утилиты](#утилиты)
14. [Документация](#документация)

---

## Обзор архитектуры

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ                                │
│  ComfyUI (Efficient Loader → KSampler → FaceDetailer → Upscale → Save)     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     МНОГОУРОВНЕВЫЙ ОТБОР (Tournament)                       │
│                                                                              │
│   Level 1: группы по 5 изображений → 1 победитель                           │
│   Level 2: победители Level 1 (по 5) → 1 победитель                         │
│   Level 3: победители Level 2 (по 5) → финалист                             │
│   Level 4: финалист → сохраняется в retrain_dir                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     РАНЖИРОВАНИЕ (ResNet50 + MLP Head)                      │
│                                                                              │
│   Input: 5 изображений (360x360)                                            │
│   Backbone: ResNet50 (Danbooru pretrained, 6000 tags)                       │
│   Head: 4096 → 2048 → 1024 → 512 → 256 → 1                                  │
│   Output: score для каждого изображения                                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM МОДУЛИ (опционально)                                │
│                                                                              │
│   • vlm.py — генерация анимационных промптов из изображения                │
│   • llm/llm2_model.py — генерация тегов (Transformer)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Структура проекта

```
Модели обучения (ранжирование)
├── new_top_test2.py         # Основной скрипт: Config class, AMP, K-Fold, train/val/retrain
├── retrain.py               # Дообучение на новых данных
├── new_top.py               # SoftFocalPairwiseLoss
├── new_top_test.py          # SoftFocalPairwiseLoss + SEBlock
├── listwise.py              # Listwise + Pairwise loss
├── test.py                  # ResNet + SentenceTransformer
├── test33.py                # ReduceLROnPlateau + validation loss
│
Эксперименты (K-Fold)
├── test 7036 3725.py        # ListNet + RankNet
├── test 7036 3883.py        # LambdaLoss (NDCG-based)
├── test 7046 3883.py        # ListNet + RankNet (LayerNorm)
├── test 7118 4112.py        # Top1Probability + Pairwise
├── test 7177 4299.py        # Top1Probability + Pairwise
│
Визуализация
├── vizualize.py             # t-SNE проекция эмбеддингов
├── vizualize2.py            # Расширенная статистика и визуализация
│
Отбор изображений
├── predict.py               # Отбор лучших (4-level tournament)
│
Генерация и API
├── vlm.py                   # VLM для анимационных промптов
├── danbooru_resnet.py       # ResNet50 теггер (6000 тегов)
├── test_resnet50_danbooru.py# Тест Danbooru теггера
├── comfyui_workflow.py      # Полный автоматический пайплайн
│
Danbooru API
├── danbooru_api/
│   ├── main.py              # Сбор постов с прокси
│   ├── __init__.py          # Python package
│   ├── character_tags.txt   # Разрешённые персонажи
│   ├── skip_tags.txt        # Теги для пропуска
│   ├── remove_tags.txt      # Теги для удаления
│   ├── new_tags.txt         # Новые найденные теги
│   └── danbooru_posts.db    # SQLite база
│
LLM модули
├── llm/
│   ├── llm2_model.py        # Transformer генератор тегов
│   ├── llm2_dataset.py      # CSV → уникальные промпты
│   ├── parse_dataset.py     # Парсинг CSV (Danbooru API)
│   ├── shuffle_dataset.py   # Перемешивание + Zipf/Jaccard анализ
│   └── tag_coverage.py      # Анализ покрытия тегов
│
Workflows
├── base.json                # Генерация аниме (ComfyUI)
├── ez.json                  # Wan2.1 I2V анимация
├── class_names_6000.json    # 6000 Danbooru тегов
│
Утилиты
├── meanstd.py               # Вычисление mean/std
├── util_clear_metadata.py   # Очистка метаданных PNG
├── sdfsdf.py                # SQLite утилиты
├── util test.py             # Сброс autoincrement
├── train.py                 # Очистка кэша HuggingFace
├── train.txt / val.txt      # Сплиты (set_XXX)
└── dataset/                 # Обучающие данные
    ├── train/               # train группы
    ├── val/                 # val группы
    └── retrain/             # дообучение
```

---

## Требования

```bash
pip install torch torchvision albumentations pillow numpy scikit-learn tqdm
pip install transformers bitsandbytes
pip install matplotlib pandas plotly
```

---

## Быстрый старт

### 1. Генерация изображений (ComfyUI)

Используйте `base.json` как шаблон workflow в ComfyUI.

### 2. Визуализация эмбеддингов

```bash
python vizualize.py  # t-SNE проекция с подсветкой лучших изображений
```

### 3. Отбор лучших изображений

```python
from predict import select_best_4level_flat

select_best_4level_flat(
    model_path="best_model.pth",
    input_dir=r"D:\ComfyUI\output\dataset",
    group_size=5,
    batch_size=125,
    save_threshold=2.0,
    output_dir=r"D:\finish",
    copy_from_dir=r"D:\ComfyUI\output"
)
```

### 4. Обучение модели

```python
# Основной вариант (K-Fold, AMP, Config class)
from new_top_test2 import train
train()

# Дообучение на новых данных
from retrain import retrain
retrain()

# Базовый вариант (pairwise)
from new_top import train
train()

# Комбинированный лосс
from listwise import train
train()
```

### 5. VLM генерация промптов

```python
from vlm import generate_animation_prompt
prompt = generate_animation_prompt("image.png")
```

### 6. LLM генерация тегов

```bash
python llm/llm2_dataset.py
python llm/shuffle_dataset.py  # включает Zipf + Jaccard анализ
python llm/llm2_model.py
```

### 7. Сбор данных с Danbooru

```python
# Вариант 1: по тегам
from danbooru_api.main import main
main(search_type="character", tag="sailor moon", rating="general")

# Вариант 2: последовательный парсинг
python llm/parse_dataset.py
```

### 8. Утилиты

```bash
# Вычисление статистик датасета
python meanstd.py

# Очистка метаданных PNG
python util_clear_metadata.py "D:\finish" --backup
```

---

## Формат данных

### Группа изображений

```
dataset/
└── set_001/
    ├── 001.png
    ├── 002.png
    ├── 003.png
    ├── 004.png
    ├── 005.png
    └── best.txt          # "003.png"
```

### train.txt / val.txt

Списки групп для обучения:
```
set_001
set_002
set_003
...
```

---

## Архитектура модели

### EnhancedAnimeRanker (базовая)

```
Input: [B, 5, 3, 360, 360]

ResNet50 (Danbooru pretrained)
└── Backbone: [B*5, 4096]

MLP Head:
├── Linear(4096, 2048) + GELU + LayerNorm + Dropout(0.3)
├── Linear(2048, 1024) + GELU + LayerNorm + Dropout(0.2)
├── Linear(1024, 512)  + GELU + LayerNorm + Dropout(0.2)
├── Linear(512, 256)   + GELU + Dropout(0.1)
└── Linear(256, 1)

Output: [B, 5] scores
```

### EnhancedAnimeRanker + Text (test.py)

```
Input: [B, 5, 3, 360, 360] + [B, 5, 384] text embeddings
Concat: [B*5, 4480]
MLP Head: 4480 → ... → 1
```

---

## Варианты обучения

| Файл | Подход | Loss | Особенности |
|------|--------|------|-------------|
| `new_top_test2.py` | Pairwise | SoftFocalPairwiseLoss | **Основной**: Config class, AMP, K-Fold, train/val/retrain |
| `retrain.py` | Дообучение | SoftFocalPairwiseLoss | Классический скрипт дообучения |
| `new_top.py` | Pairwise | SoftFocalPairwiseLoss | Базовая версия |
| `new_top_test.py` | Pairwise + SEBlock | SoftFocalPairwiseLoss | С attention механизмом |
| `listwise.py` | Listwise + Pairwise | ListwiseLoss + PairwiseMarginLoss | Комбинированный лосс |
| `test.py` | Pairwise + Text | PairwiseRankingLoss | С текстовыми эмбеддингами |
| `test33.py` | Pairwise | SoftFocalPairwiseLoss | ReduceLROnPlateau + validation loss |
| `test 7036 3725.py` | K-Fold 5 | ListNetLoss + RankNetLoss | |
| `test 7036 3883.py` | K-Fold 5 | LambdaLoss | NDCG-based |
| `test 7046 3883.py` | K-Fold 5 | ListNetLoss + RankNetLoss | С LayerNorm |
| `test 7118 4112.py` | K-Fold 5 | Top1Probability + Pairwise | |
| `test 7177 4299.py` | K-Fold 5 | Top1Probability + Pairwise | |

### K-Fold Cross-Validation

```
train_val (90%) ──┬── Fold 1: train[2,3,4,5], val[1]
                  ├── Fold 2: train[1,3,4,5], val[2]
                  ├── Fold 3: train[1,2,4,5], val[3]
                  ├── Fold 4: train[1,2,3,5], val[4]
                  └── Fold 5: train[1,2,3,4], val[5]

Final: train_val → test (10%)
```

---

## Loss Функции

### SoftFocalPairwiseLoss

```python
diff = target * (scores_best - scores_other) / temperature
pt = torch.sigmoid(diff)
focal_weight = (1 - pt).gamma
loss = alpha * focal_weight * torch.log1p(torch.exp(margin - diff))
```

### RankNetLoss

```python
diff = pos_score - neg_score - margin
loss = -torch.log(sigmoid(diff))
```

### ListNetLoss

```python
target_dist[target] = 1.0 - label_smoothing
loss = -sum(target_dist * log(softmax(scores)))
```

### LambdaLoss

NDCG-based ranking loss — оптимизирует градиенты NDCG напрямую.

### Top1ProbabilityLoss

```python
top1_loss = -mean(target_prob * log(top1_prob) + (1-target_prob) * log(1-top1_prob))
```

### CombinedLoss

```python
loss = top1_weight * top1_loss + pairwise_weight * pairwise_loss
```

---

## Метрики

| Метрика | Описание |
|---------|----------|
| **NDCG@5** | Normalized Discounted Cumulative Gain |
| **Top-1 Accuracy** | Правильный выбор лучшего |
| **Top-2 Accuracy** | Лучшее в топ-2 |

---

## Конфигурация

| Параметр | Описание | Значение |
|----------|----------|----------|
| `img_size` | Размер изображения | 224 (training) / 360 (validation) |
| `batch_size` | Размер батча | 32 |
| `head_lr` | LR для головы | 1e-4 / 5e-5 |
| `backbone_lr` | LR для backbone | 1e-6 / 1e-5 |
| `epochs` | Эпох | 150 |
| `n_folds` | K-Fold | 5 |
| `test_size` | Test сплит | 0.1 |
| `weight_decay` | L2 регуляризация | 0.05 |
| `grad_clip` | Градиентный клиппинг | 0.5 |

---

## ComfyUI Workflows

### base.json — Генерация аниме

```
[Efficient Loader] ──┬── [KSampler] ──> [FaceDetailer eyes] ──> [FaceDetailer face] ──> [Upscale] ──> [Save]
                     └── [VAE] ─────────────────────────────────────────────────────────────────────┘
```

- LoRA: sparklecolor10.5
- FaceDetailer: SAM + YOLO
- Upscale: 4x-AnimeSharp

### ez.json — Анимация (Wan2.1 I2V)

```
[LoadImage] ──> [WanFirstLastFrameToVideo] ──> [KSamplerAdvanced x2] ──> [VAEDecode] ──> [VideoCombine (GIF)]
```

- 16fps выход
- LoRA: Wan2.2-Lightning

---

## База данных

### danbooru_posts.db

```sql
CREATE TABLE requests (
    request_id INTEGER PRIMARY KEY,
    user_tag_formatted TEXT
);

CREATE TABLE posts (
    post_id INTEGER PRIMARY KEY,
    tags TEXT,
    rating_category TEXT,  -- 'sfw' или 'nsfw'
    request_id INTEGER
);
```

---

## Утилиты

| Скрипт | Назначение |
|--------|------------|
| `vizualize.py` | t-SNE визуализация эмбеддингов |
| `vizualize2.py` | Расширенная визуализация и статистика |
| `meanstd.py` | Вычисление mean/std для нормализации |
| `util_clear_metadata.py` | Очистка метаданных из PNG |
| `sdfsdf.py` | Работа с SQLite |
| `util test.py` | Сброс autoincrement |
| `comfyui_workflow.py` | Полный пайплайн: тег → генерация → отбор → анимация |

### Автоматический пайплайн (comfyui_workflow.py)

```
1. Получить тег с Danbooru
2. Собрать 25 постов
3. Сгенерировать 125 изображений (5 прогонов)
4. Отобрать лучшие (4-level tournament)
5. Сгенерировать анимацию (Wan2.1 I2V)
6. Сжать GIF (gifsicle)
7. Запустить SafeVision
8. Добавить в архив
9. Выключить ПК
```

---

## Документация

### Курсовая/дипломная работа

**Овчаренко К.С. ЭФБО-04-23 Разработка моделей для генерации промптов и ранжирования изображений.docx**

Подробное описание проекта включает:

#### Теоретическая часть

**Модель генерации промптов:**
- Сбор данных с Danbooru API с фильтрацией по разрешённым персонажам
- Предобработка: двойное перемешивание, отсечение топ-1000 тегов
- Анализ по закону Ципфа (s = -1.1151) и матрица совместной встречаемости (Jaccard)
- Mixup-аугментация для улучшения обобщения
- Архитектура: decoder-only трансформер с нуля (D_MODEL=512, NHEAD=8, NUM_LAYERS=4)

**Модель ранжирования изображений:**
- Датасет: группы по 5 изображений с разметкой лучшего
- Pairwise-подход с кастомным SoftFocalPairwiseLoss
- Backbone: ResNet-50, предобученная на Danbooru
- Метрика: NDCG@5, Top-1/Top-2 Accuracy

#### Практическая часть

- Реализация сбора данных: 158 256 → 143 714 уникальных записей
- Обучение модели промптов: 120 эпох с Mixup-аугментацией
- Полный разбор кода с листингами функций

---

## Лицензия

Проект для личного использования.