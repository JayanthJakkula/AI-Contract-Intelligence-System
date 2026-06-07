import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import BaselineModel, AttentionModel

# Set random seed for PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class ClauseDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        # Map labels to integers
        self.label_map = {
            "Confidentiality Clause": 0,
            "Termination Clause": 1,
            "Payment Clause": 2,
            "Liability Clause": 3,
            "Non-Compete Clause": 4
        }
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(item["padded_indices"], dtype=torch.long)
        y = torch.tensor(self.label_map[item["label"]], dtype=torch.long)
        return x, y

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15, is_attention=False, device="cpu"):
    model.to(device)
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            if is_attention:
                logits, _ = model(x_batch)
            else:
                logits = model(x_batch)
                
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, is_attention, device)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

def evaluate_model(model, loader, is_attention=False, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            if is_attention:
                logits, _ = model(x_batch)
            else:
                logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset and vocab
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/preprocessed_data.json", "r", encoding='utf-8') as f:
        preprocessed_data = json.load(f)
        
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/vocab.json", "r", encoding='utf-8') as f:
        vocab = json.load(f)
        
    vocab_size = len(vocab)
    embedding_dim = 64
    num_heads = 4
    num_classes = 5
    max_len = 50
    epochs = 15
    batch_size = 32
    
    # Group data by subset
    train_data = [d for d in preprocessed_data if d["subset"] == "train"]
    val_data = [d for d in preprocessed_data if d["subset"] == "dev"]
    test_data = [d for d in preprocessed_data if d["subset"] == "test"]
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")
    
    # Create dataloaders
    train_dataset = ClauseDataset(train_data)
    val_dataset = ClauseDataset(val_data)
    test_dataset = ClauseDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Train Baseline Model ---
    print("\n--- Training Baseline Model (Embedding + Dense) ---")
    baseline_model = BaselineModel(vocab_size, embedding_dim, num_classes)
    optimizer_b = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_model(baseline_model, train_loader, val_loader, criterion, optimizer_b, epochs, is_attention=False, device=device)
    
    test_acc_b, test_prec_b, test_rec_b, test_f1_b = evaluate_model(baseline_model, test_loader, is_attention=False, device=device)
    print("\nBaseline Model Test Metrics:")
    print(f"  Accuracy:  {test_acc_b:.4f}")
    print(f"  Precision: {test_prec_b:.4f}")
    print(f"  Recall:    {test_rec_b:.4f}")
    print(f"  F1 Score:  {test_f1_b:.4f}")
    
    # Save baseline weights
    torch.save(baseline_model.state_dict(), "d:/Deep_Learning/AI Contract Intelligence System/data/baseline_model.pt")
    
    # --- Train Self-Attention Model ---
    print("\n--- Training Self-Attention Model (Embedding + PE + Attention + Dense) ---")
    attention_model = AttentionModel(vocab_size, embedding_dim, num_heads, num_classes, max_len=max_len)
    optimizer_a = optim.Adam(attention_model.parameters(), lr=0.001)
    
    train_model(attention_model, train_loader, val_loader, criterion, optimizer_a, epochs, is_attention=True, device=device)
    
    test_acc_a, test_prec_a, test_rec_a, test_f1_a = evaluate_model(attention_model, test_loader, is_attention=True, device=device)
    print("\nSelf-Attention Model Test Metrics:")
    print(f"  Accuracy:  {test_acc_a:.4f}")
    print(f"  Precision: {test_prec_a:.4f}")
    print(f"  Recall:    {test_rec_a:.4f}")
    print(f"  F1 Score:  {test_f1_a:.4f}")
    
    # Save attention weights
    torch.save(attention_model.state_dict(), "d:/Deep_Learning/AI Contract Intelligence System/data/attention_model.pt")
    
    # Save model comparison stats
    comparison = {
        "baseline": {
            "accuracy": test_acc_b,
            "precision": test_prec_b,
            "recall": test_rec_b,
            "f1": test_f1_b
        },
        "attention": {
            "accuracy": test_acc_a,
            "precision": test_prec_a,
            "recall": test_rec_a,
            "f1": test_f1_a
        }
    }
    
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/model_comparison.json", "w") as cf:
        json.dump(comparison, cf, indent=2)
    print("\nSaved model comparisons to data/model_comparison.json")

if __name__ == "__main__":
    main()
