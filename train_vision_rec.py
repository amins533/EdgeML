import os
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- PATH CONFIGURATIONS ---
BASE_DIR = Path(__file__).resolve().parent
CARDS_DIR = BASE_DIR / "cards"
CSV_BREAKDOWN_PATH = BASE_DIR / "breakdown-2024.csv"

def parse_clip_index(filename_str):
    # Extracts numbers out of filenames like 'clip_42_playcard.png' or '42.mp4'
    digits = ''.join(filter(str.isdigit, str(filename_str)))
    return int(digits) if digits else None

def get_breakdown_mapping():
    if not CSV_BREAKDOWN_PATH.exists(): 
        print(f"[ERROR] Could not find play breakdown sheet at: {CSV_BREAKDOWN_PATH}")
        return {}
    try:
        with open(CSV_BREAKDOWN_PATH, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
        df = pd.read_csv(CSV_BREAKDOWN_PATH, sep=delimiter)
    except Exception as e:
        print(f"[ERROR] Reading breakdown spreadsheet failed: {e}")
        return {}
    
    play_family_map = {
        'DIVE': 'Inside Run', 'POWER': 'Inside Run', 'DUO': 'Inside Run', 
        'COUNTER': 'Inside Run', 'TRAP': 'Inside Run', 'BLAST': 'Inside Run', 'SNEAK': 'Inside Run',
        'STRETCH': 'Outside Run', 'JET SWEEP': 'Outside Run',
        'LUNCH': 'Quick Pass', 'BREAKFAST': 'Quick Pass', 'ARROW': 'Quick Pass', 
        'CIGARETTE': 'Quick Pass', 'SLANT': 'Quick Pass', 'STICK': 'Quick Pass', 
        'SLIP': 'Quick Pass', 'JAILBREAK': 'Quick Pass', 'BUBBLE': 'Quick Pass',
        'DINNER': 'Dropback Pass', 'SUNDAY': 'Dropback Pass', 'VERTS': 'Dropback Pass', 
        'CURL': 'Dropback Pass', 'OUT': 'Dropback Pass', 'FADE': 'Dropback Pass', 
        'DRIVE': 'Dropback Pass', 'SMASH': 'Dropback Pass', 'WHEEL': 'Dropback Pass'
    }

    mapping = {}
    for idx, row in df.iterrows():
        clip_num = idx + 1
        off_play_val = str(row.get('OFF PLAY', '')).strip().upper()
        off_form_val = str(row.get('OFF FORM', '')).strip().upper()
        raw_label = off_play_val if off_play_val and off_play_val != 'NAN' else off_form_val
        if not raw_label or raw_label == 'NAN': continue
        resolved_category = play_family_map.get(raw_label, None)
        if resolved_category:
            mapping[clip_num] = resolved_category
    return mapping

# --- PYTORCH PLAY CARD CUSTOM DATASET ---
class PlayCardDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    print("==================================================")
    print("🏈       DEEP LEARNING VISION PLAY RECOGNITION    🏈")
    print("==================================================")

    if not CARDS_DIR.exists():
        print(f"[ERROR] Image repository folder not found at '{CARDS_DIR}'. Run main.py first to generate play cards.")
        return

    excel_map = get_breakdown_mapping()
    
    all_image_paths = []
    all_labels = []

    # Match all generated images in cards/ folder with labels
    for file in CARDS_DIR.glob("*_playcard.png"):
        clip_num = parse_clip_index(file.name)
        if clip_num in excel_map:
            all_image_paths.append(str(file))
            all_labels.append(excel_map[clip_num])

    if not all_image_paths:
        print("[ERROR] No generated play card images could be mapped back to your breakdown spreadsheet labels.")
        return

    print(f"[INFO] Successfully indexed {len(all_image_paths)} play card images across the canvas.")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    num_classes = len(label_encoder.classes_)

    # Stratified Split
    X_train_p, X_val_p, y_train_idx, y_val_idx = train_test_split(
        all_image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Image Augmentations to prevent overfitting on small image sets
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PlayCardDataset(X_train_p, y_train_idx, transform=train_transforms)
    val_dataset = PlayCardDataset(X_val_p, y_val_idx, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Hardware Acceleration Hook
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Directing neural calculations to active hardware device: {device.type.upper()}")

    # Initialize a deep learning vision network (ResNet-18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Swap final dense classification head
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    epochs = 15
    print(f"[TRAINING] Optimization loop started for {epochs} training epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"  -> Epoch [{epoch+1:02d}/{epochs}] Complete | Training CrossEntropy Loss: {epoch_loss:.4f}")

    # --- EVALUATION ---
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    print("\n" + "="*50)
    print("🎯    VISION PLAY CARD ENGINE PERFORMANCE REPORT     ")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_, zero_division=0))

    print("="*50)
    print("🧱                  CONFUSION MATRIX               ")
    print("="*50)
    cm = confusion_matrix(all_targets, all_preds)
    header = f"{'True \\ Pred':<15}" + "".join([f"{c[:10]:>12}" for c in label_encoder.classes_])
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(cm):
        row_str = f"{label_encoder.classes_[idx]:<15}" + "".join([f"{val:>12}" for val in row])
        print(row_str)

    # Save weights
    torch.save(model.state_dict(), "play_vision_model.pth")
    print("\n💾 Saved deep vision network states to 'play_vision_model.pth'")
    print(f"🚀 [SUMMARY] PLAY CARD MODEL ACCURACY: {accuracy_score(all_targets, all_preds) * 100:.2f}%\n")

if __name__ == "__main__":
    main()