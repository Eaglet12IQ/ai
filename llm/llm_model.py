import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

# Фиксация случайных чисел для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Проверка наличия файлов
if not os.path.exists('prompt_dataset.csv'):
    raise FileNotFoundError("File 'prompt_dataset.csv' not found. Please ensure it exists in the working directory.")

# Загрузка датасета
try:
    df = pd.read_csv('prompt_dataset.csv')
except Exception as e:
    raise Exception(f"Error loading dataset: {e}")

# Проверка столбцов и их содержимого
required_columns = ['input', 'output']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")
df = df.dropna(subset=required_columns)  # Удаление строк с NaN

# Проверка, что датасет не пустой
if df.empty:
    raise ValueError("Dataset is empty. Please check 'prompt_dataset.csv'.")

# Построение словаря тегов
tags_file = 'all_tags.pkl'
if os.path.exists(tags_file):
    with open(tags_file, 'rb') as f:
        all_tags = pickle.load(f)
else:
    all_tags = set()
    for prompt in df['output']:
        tags = [tag.strip() for tag in prompt.split(',')]
        all_tags.update(tags)
    for input_tag in df['input']:
        all_tags.add(input_tag.strip())
    all_tags = list(all_tags)
    with open(tags_file, 'wb') as f:
        pickle.dump(all_tags, f)
all_tags_set = set(all_tags)  # Для быстрого поиска

if not all_tags:
    raise ValueError("No tags found in dataset. Please check the 'output' and 'input' columns.")

# Аугментация: случайное перемешивание, добавление и удаление тегов
def augment_prompt(prompt, input_tag=None, max_add_tags=2, max_remove_tags=1):
    tags = [tag.strip() for tag in prompt.split(',')]
    random.shuffle(tags)  # Перемешивание тегов
    if input_tag and input_tag not in tags:
        tags.insert(0, input_tag)
    return ', '.join(tags)

# Кастомный датасет
class PromptDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=128):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"generate prompt: {self.inputs[idx]}"
        output_text = augment_prompt(self.outputs[idx], self.inputs[idx]) if random.random() > 0.5 else self.outputs[idx]
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze()
        }

# Разделение данных
train_df, val_df = train_test_split(df, test_size=0.001, random_state=42)

# Инициализация токенизатора и модели
try:
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
except Exception as e:
    raise Exception(f"Error loading T5 model or tokenizer: {e}")

# Подготовка датасетов
train_dataset = PromptDataset(train_df['input'].values, train_df['output'].values, tokenizer)
val_dataset = PromptDataset(val_df['input'].values, val_df['output'].values, tokenizer)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir='./t5_prompt_model',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='steps',
    save_strategy='steps',
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='loss',  # Используем loss вместо метрик
    greater_is_better=False,       # Меньший loss лучше
    fp16=torch.cuda.is_available(),
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics удалено
)

# Обучение модели
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Сохранение модели и токенизатора
trainer.save_model('./t5_prompt_model')
model.save_pretrained('./t5_prompt_model')
tokenizer.save_pretrained('./t5_prompt_model')

# Генерация уникального промпта
def generate_unique_prompt(input_word, model, tokenizer, max_length=50):
    model.eval()
    input_text = f"generate prompt: {input_word}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=150,
        top_p=0.85,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tags = [tag.strip() for tag in generated.split(',')]
    valid_tags = [tag for tag in tags if tag in all_tags_set]
    return ', '.join(valid_tags)

# Тестирование генерации
try:
    model = T5ForConditionalGeneration.from_pretrained('./t5_prompt_model')
    tokenizer = T5Tokenizer.from_pretrained('./t5_prompt_model')
    test_words = ['rio \\(blue archive\\)', '1girl', 'breasts', 'long hair', 'frieren']
    for word in test_words:
        print(f"Input: {word}")
        for i in range(3):  # Генерируем 3 варианта для разнообразия
            prompt = generate_unique_prompt(word, model, tokenizer)
            print(f"Generated Prompt {i+1}: {prompt}")
except Exception as e:
    print(f"Error during generation: {e}")