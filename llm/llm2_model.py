import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import math
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

class Config:
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATA_PATH = 'danbooru_general_unique_shuffled_cleaned_top_1000.csv'
    MAX_TAGS = 42
    TEST_SIZE = 0.1
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 120
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 10
    MIX_PROB = 0.25
    MIX_ALPHA = 0.4
    
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.2
    MAX_LEN = 1024
    
    TEMPERATURE = 0.9
    TOP_K = 40
    TOP_P = 0.9
    REPETITION_PENALTY = 1.6
    FREQUENCY_PENALTY_STRENGTH = 2.0
    MAX_NEW_TOKENS = 60


def set_seed(seed: int = Config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def load_and_preprocess_data(data_path: str, device: torch.device):
    df = pd.read_csv(data_path)
    sequences = [row['tags'].strip().split(', ') for _, row in df.iterrows() if row['tags'].strip()]
    
    EOS_TOKEN = "<EOS>"
    sequences = [seq + [EOS_TOKEN] for seq in sequences]

    all_tags = [tag for seq in sequences for tag in seq]
    counter = Counter(all_tags)
    vocab = ["<PAD>", "<EOS>"] + [t for t, _ in counter.most_common() if t not in ("<PAD>", "<EOS>")]
    
    tag_to_idx = {tag: idx for idx, tag in enumerate(vocab)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    byte_length = torch.zeros(len(vocab), dtype=torch.float32, device=device)
    for idx, tag in idx_to_tag.items():
        if tag not in ("<PAD>", "<EOS>"):
            byte_length[idx] = len(tag.encode('utf-8'))
    
    indexed_sequences = [[tag_to_idx[tag] for tag in seq] for seq in sequences]
    
    return indexed_sequences, tag_to_idx, idx_to_tag, byte_length


def augment_sequence(
    seq: List[int],
    max_tags: int = Config.MAX_TAGS,
    mix_prob: float = 0.0,
    mix_alpha: float = 0.4,
    mix_source: Optional[List[List[int]]] = None,
    shuffle: bool = True
) -> List[int]:
    tags = seq[:]
    
    if mix_prob > 0 and mix_source and random.random() < mix_prob:
        seq_b = random.choice(mix_source)
        lam = np.random.beta(mix_alpha, mix_alpha) if mix_alpha > 0 else 0.5
        n_take = max(1, int(len(seq_b) * lam))
        
        space_left = max_tags - len(tags)
        if space_left > 0:
            added = random.sample(seq_b, min(n_take, len(seq_b), space_left))
            combined = list(dict.fromkeys(tags + added))
            tags = combined[:max_tags]
    
    if shuffle:
        random.shuffle(tags)
    
    return tags


class TagDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
        tag_to_idx: Dict[str, int],
        shuffle_tags: bool = True,
        mix_prob: float = 0.0,
        mix_alpha: float = 0.4,
        mix_source: Optional[List[List[int]]] = None,
        max_tags: int = Config.MAX_TAGS
    ):
        self.sequences = sequences
        self.tag_to_idx = tag_to_idx
        self.shuffle_tags = shuffle_tags
        self.mix_prob = mix_prob
        self.mix_alpha = mix_alpha
        self.mix_source = mix_source
        self.max_tags = max_tags
        self.eos_idx = tag_to_idx["<EOS>"]
        self.pad_idx = tag_to_idx["<PAD>"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        full_seq = self.sequences[idx][:]
        tags = full_seq[:-1]
        
        augmented_tags = augment_sequence(
            seq=tags,
            max_tags=self.max_tags,
            mix_prob=self.mix_prob,
            mix_alpha=self.mix_alpha,
            mix_source=self.mix_source,
            shuffle=self.shuffle_tags
        )
        
        shuffled_seq = augmented_tags + [self.eos_idx]
        input_seq = shuffled_seq[:-1]
        target_seq = shuffled_seq[1:]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)

    def __len__(self) -> int:
        return len(self.sequences)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    
    return padded_inputs, padded_targets


def create_dataloaders(
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    tag_to_idx: Dict[str, int],
    batch_size: int = Config.BATCH_SIZE
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = TagDataset(
        sequences=train_seqs,
        tag_to_idx=tag_to_idx,
        shuffle_tags=True,
        mix_prob=Config.MIX_PROB,
        mix_alpha=Config.MIX_ALPHA,
        mix_source=train_seqs,
        max_tags=Config.MAX_TAGS
    )
    
    val_dataset = TagDataset(
        sequences=val_seqs,
        tag_to_idx=tag_to_idx,
        shuffle_tags=False,
        mix_prob=0.0,
        mix_source=None,
        max_tags=Config.MAX_TAGS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = Config.MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TagTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, Config.D_MODEL, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(Config.D_MODEL)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        self.fc = nn.Linear(Config.D_MODEL, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x == self.pad_idx)
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


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                   byte_length: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_weighted_nll = 0.0
    total_bytes = 0

    for inputs, targets in tqdm(loader, desc="Validation", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        total_loss += loss.item()
        
        per_token_nll = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), 
            targets.view(-1), 
            ignore_index=-100, 
            reduction='none'
        )
        
        mask = (targets.view(-1) != -100)
        if mask.any():
            valid_targets = targets.view(-1)[mask]
            bytes_tensor = byte_length[valid_targets].to(device)
            
            total_weighted_nll += (per_token_nll[mask] * bytes_tensor).sum().item()
            total_bytes += bytes_tensor.sum().item()
    
    avg_loss = total_loss / len(loader)
    bpb = (total_weighted_nll / total_bytes) / math.log(2) if total_bytes > 0 else 0.0
    
    return avg_loss, bpb


def generate_full_prompt(
    model: nn.Module,
    start_tags: List[str],
    tag_to_idx: Dict[str, int],
    idx_to_tag: Dict[int, str],
    temperature: float = Config.TEMPERATURE,
    max_new_tokens: int = Config.MAX_NEW_TOKENS,
    device: torch.device = Config.DEVICE
) -> str:
    model.eval()
    sequence = start_tags.copy()
    used = set(start_tags)
    
    for _ in range(max_new_tokens):
        input_ids = [tag_to_idx[t] for t in sequence]
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]
            
            for tag in used:
                if tag in tag_to_idx:
                    logits[0, tag_to_idx[tag]] = -1e9
            logits[0, tag_to_idx["<PAD>"]] = -1e9
            
            probs = torch.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
        
        next_tag = idx_to_tag[next_idx]
        if next_tag == "<EOS>":
            break
            
        sequence.append(next_tag)
        used.add(next_tag)
    
    return ", ".join(sequence)


def main():
    set_seed()
    device = Config.DEVICE
    print(f"Using device: {device}")

    print("Loading and preprocessing data...")
    indexed_sequences, tag_to_idx, idx_to_tag, byte_length = load_and_preprocess_data(
        Config.DATA_PATH, device
    )
    
    train_seqs, val_seqs = train_test_split(
        indexed_sequences, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    
    train_loader, val_loader = create_dataloaders(train_seqs, val_seqs, tag_to_idx)

    vocab_size = len(tag_to_idx)
    model = TagTransformer(vocab_size, pad_idx=tag_to_idx["<PAD>"]).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...\n")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_bpb = validate_epoch(model, val_loader, criterion, byte_length, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1:2d}/{Config.NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val BPB: {val_bpb:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_tag_transformer.pth")
            print("→ Лучшая модель сохранена")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping на эпохе {epoch+1}")
                break

    print("\n" + "="*60)
    print("Генерация примеров:")
    model.load_state_dict(torch.load("best_tag_transformer.pth", map_location=device))
    
    for i in range(5):
        prompt = generate_full_prompt(
            model=model,
            start_tags=["1girl"],
            tag_to_idx=tag_to_idx,
            idx_to_tag=idx_to_tag,
            temperature=0.9,
            max_new_tokens=80
        )
        print(f"Вариант {i+1}: {prompt}")


if __name__ == "__main__":
    main()