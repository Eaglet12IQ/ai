import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import os
import re

# Фиксация случайности
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None

# ─────────────────────────────────────────────────────────────
# 1. Подготовка данных (без изменений датасета!)
# ─────────────────────────────────────────────────────────────

df = pd.read_csv('llm/unique_prompts.csv')
sequences = [row['output'].strip().split(', ') for _, row in df.iterrows() if row['output'].strip()]

# Добавляем EOS
EOS_TOKEN = "<EOS>"
sequences = [seq + [EOS_TOKEN] for seq in sequences]

# Словарь по частоте
all_tags = [tag for seq in sequences for tag in seq]
counter = Counter(all_tags)
all_tags = ["<PAD>", "<EOS>"] + [t for t, _ in counter.most_common() if t not in ["<PAD>", "<EOS>"]]

vocab_size = len(all_tags)

tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

pad_idx = tag_to_idx["<PAD>"]
eos_idx = tag_to_idx["<EOS>"]

# Индексация
indexed_sequences = [[tag_to_idx[tag] for tag in seq] for seq in sequences]

train_seqs, val_seqs = train_test_split(indexed_sequences, test_size=0.1, random_state=42)

class TagDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)

    return padded_inputs, padded_targets

batch_size = 64
train_dataset = TagDataset(train_seqs)
val_dataset   = TagDataset(val_seqs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ─────────────────────────────────────────────────────────────
# 2. Модель (без изменений)
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

def generate_square_subsequent_mask(sz: int):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TagTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.2, max_len=1024, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        padding_mask = (x == self.embedding.padding_idx)

        x = self.embedding(x)
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len).to(x.device)

        x = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        return self.fc(x)

model = TagTransformer(vocab_size, pad_idx=pad_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ─────────────────────────────────────────────────────────────
# 3. Loss и оптимизатор
# ─────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ─────────────────────────────────────────────────────────────
# 4. Обучение
# ─────────────────────────────────────────────────────────────

num_epochs = 30
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "llm/best_tag_transformer.pth")
        print("→ Сохранена лучшая модель")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break

# ─────────────────────────────────────────────────────────────
# 5. Генерация с frequency penalty и no-repeat
# ─────────────────────────────────────────────────────────────

# Глобальный frequency penalty (штраф по частоте токена)
token_freq = Counter([tag for seq in sequences for tag in seq])
max_freq = max(token_freq.values())
freq_penalty_dict = {tag_to_idx[t]: token_freq[t] / max_freq for t in token_freq}

def apply_frequency_penalty(logits, freq_dict, strength=2.0):
    for idx, p in freq_dict.items():
        logits[:, idx] -= strength * p
    return logits

def apply_repetition_penalty(logits, generated, penalty=1.6, lookback=20):
    recent = set(generated[-lookback:])
    for idx in recent:
        logits[:, idx] /= penalty
    return logits

def ban_tokens(logits, banned_idxs):
    logits[:, banned_idxs] = -1e9
    return logits

def sample_token(logits, temperature=1.0, top_k=40, top_p=0.9):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        probs = sorted_probs.gather(1, sorted_indices)

    probs = probs / probs.sum(dim=-1, keepdim=True) + 1e-8
    return torch.multinomial(probs, num_samples=1).item()

def generate_full_prompt(model, start_tags, temperature=0.95, max_new_tokens=60):
    model.eval()
    
    sequence = start_tags.copy()           # список строк
    used = set(start_tags)                 # уже использованные теги (строки)
    
    banned_idxs = {tag_to_idx["<PAD>"]}
    
    for _ in range(max_new_tokens):
        # готовим вход
        input_ids = [tag_to_idx[t] for t in sequence]
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]     # shape [1, vocab]
            
            # ─── Самое главное: убираем уже использованные теги ────────
            for tag in used:
                idx = tag_to_idx.get(tag)
                if idx is not None:
                    logits[0, idx] = -1e9
            
            # технические тоже
            for idx in banned_idxs:
                logits[0, idx] = -1e9
            
            # можно оставить небольшой штраф на brown hair, если он всё равно доминирует
            # brown_idx = tag_to_idx.get("brown hair")
            # if brown_idx is not None and len(sequence) > 3:
            #     logits[0, brown_idx] -= 8.0
            
            # сэмплинг
            probs = torch.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
        
        next_tag = idx_to_tag[next_idx]
        
        if next_tag == "<EOS>":
            break
        
        sequence.append(next_tag)
        used.add(next_tag)                     # сразу запрещаем на следующий шаг
    
    return ", ".join(sequence)

# Тестирование
model.load_state_dict(torch.load("llm/best_tag_transformer.pth"))
model.eval()

print("Генерация 5 вариантов:")
for i in range(5):
    prompt = generate_full_prompt(
        model,
        start_tags=["1girl", "cosplay"],
        temperature=0.9,           # 0.8–1.1 обычно нормально
        max_new_tokens=500
    )
    print(f"Вариант {i+1}: {prompt}")