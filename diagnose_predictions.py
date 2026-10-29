import sqlite3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "plays.db"
CSV_BREAKDOWN_PATH = BASE_DIR / "breakdown-2024.csv" 

# Re-use your breakdown mapping functions from play_rec.py
def parse_clip_index(clip_id_str):
    if pd.isna(clip_id_str): return None
    digits = ''.join(filter(str.isdigit, str(clip_id_str)))
    return int(digits) if digits else None

def get_breakdown_mapping():
    if not CSV_BREAKDOWN_PATH.exists(): return {}
    delimiter = ';'
    try:
        with open(CSV_BREAKDOWN_PATH, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if ';' not in first_line and ',' in first_line: delimiter = ','
        df = pd.read_csv(CSV_BREAKDOWN_PATH, sep=delimiter)
    except: return {}
    
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
        if resolved_category: mapping[clip_num] = {'label': resolved_category}
    return mapping

# Dummy extraction matching feature shape of robust cloud system
excel_map = get_breakdown_mapping()
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT clip_id FROM play_trajectories")
rows = cursor.fetchall()
conn.close()

labels = []
for (clip_id,) in rows:
    num = parse_clip_index(clip_id)
    if num in excel_map:
        labels.append(excel_map[num]['label'])

print("=== DATASET LABEL DISTRIBUTION ===")
total = len(labels)
for k, v in Counter(labels).items():
    print(f"🏈 {k}: {v} plays ({v/total*100:.2f}%)")