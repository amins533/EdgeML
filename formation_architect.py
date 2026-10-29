import sqlite3
import numpy as np
import pandas as pd
import json
import os
import re
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import xgboost as xgb

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tactical_system.db"
CSV_BREAKDOWN_PATH = BASE_DIR / "breakdown-2024.csv"
DATASET_EXPORT_PATH = BASE_DIR / "formation_dataset.parquet"

class TacticalDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.create_curated_table()

    def create_curated_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS formations_curated (
                clip_id TEXT PRIMARY KEY,
                label TEXT,
                canonical_points TEXT
            )
        """)
        self.conn.commit()

class FormationExtractor:
    """Handles temporal clustering, invariant normalization and mapping."""
    def __init__(self, db: TacticalDatabase):
        self.db = db

    def get_breakdown_mapping(self):
        """Maps each CSV row strictly to its clip number (Row 1 = Clip 1)."""
        if not CSV_BREAKDOWN_PATH.exists():
            print(f"[ERROR] Breakdown file not found at {CSV_BREAKDOWN_PATH}")
            return {}
            
        df = pd.read_csv(CSV_BREAKDOWN_PATH, sep=';', on_bad_lines='skip')
        
        if 'OFF FORM' not in df.columns:
            print("[ERROR] Column 'OFF FORM' not found in CSV.")
            return {}

        mapping = {}
        for clip_num, (_, row) in enumerate(df.iterrows(), start=1):
            formation = str(row['OFF FORM']).strip()
            if formation.lower() != 'nan' and formation != '':
                mapping[clip_num] = formation
            
        print(f"[INFO] Successfully mapped {len(mapping)} formations based on CSV row indices.")
        return mapping

    def extract_all_formations(self):
        labels_map = self.get_breakdown_mapping()
        if not labels_map: return

        df_frames = pd.read_sql("SELECT DISTINCT clip_id FROM tracking_frames", self.db.conn)
        clip_ids = df_frames['clip_id'].tolist()
        
        processed_count = 0

        for clip_id in clip_ids:
            digits = ''.join(filter(str.isdigit, clip_id))
            if not digits: continue
            
            clip_num = int(digits)
            if clip_num not in labels_map: continue
            label = labels_map[clip_num]

            clip_df = pd.read_sql(f"SELECT * FROM tracking_frames WHERE clip_id='{clip_id}' ORDER BY frame_idx", self.db.conn)
            if len(clip_df) < 30: continue

            # Sliding window of 30 frames to find lowest variance pre-snap state
            min_var = float('inf')
            best_window = None
            
            for i in range(len(clip_df) - 30 + 1):
                window = clip_df.iloc[i:i+30]
                centroids_x, centroids_y = [], []
                for _, r in window.iterrows():
                    pts = json.loads(r['offense_coords'])
                    if pts:
                        centroids_x.append(np.mean([p[0] for p in pts]))
                        centroids_y.append(np.mean([p[1] for p in pts]))
                
                if centroids_x:
                    var = np.var(centroids_x) + np.var(centroids_y)
                    if var < min_var:
                        min_var = var
                        best_window = window

            if best_window is None: continue

            # Aggregate all detected coordinates across the stable window
            x_off, x_def, x_center, x_qb = [], [], [], []
            for _, r in best_window.iterrows():
                x_off.extend(json.loads(r['offense_coords']))
                x_def.extend(json.loads(r['defense_coords']))
                x_center.extend(json.loads(r['center_coords']))
                x_qb.extend(json.loads(r['qb_coords']))

            x_all_offense = x_off + x_center + x_qb
            if len(x_all_offense) < 11: continue 

            # Standardize attack direction (always attack Left -> Right)
            mean_def_x = np.mean([p[0] for p in x_def]) if x_def else 0
            mean_off_x = np.mean([p[0] for p in x_all_offense])
            
            x_all_offense = np.array(x_all_offense, dtype=float)
            if mean_def_x < mean_off_x and mean_def_x != 0:
                x_all_offense[:, 0] = 1200.0 - x_all_offense[:, 0]
                if x_center: x_center = [[1200.0 - p[0], p[1]] for p in x_center]
                if x_qb: x_qb = [[1200.0 - p[0], p[1]] for p in x_qb]

            # Extract exact 11 canonical positions using KMeans over accumulated points
            kmeans = KMeans(n_clusters=11, random_state=42, n_init=10).fit(x_all_offense)
            centroids = kmeans.cluster_centers_

            # Target Center and QB centroids
            if len(x_center) > 0:
                yolo_center_mean = np.mean(x_center, axis=0)
                center_idx = np.argmin(np.linalg.norm(centroids - yolo_center_mean, axis=1))
            else:
                center_idx = np.argmax(centroids[:, 0]) 

            center_pt = centroids[center_idx]

            if len(x_qb) > 0:
                yolo_qb_mean = np.mean(x_qb, axis=0)
                dists_to_qb = np.linalg.norm(centroids - yolo_qb_mean, axis=1)
                dists_to_qb[center_idx] = np.inf 
                qb_idx = np.argmin(dists_to_qb)
            else:
                fallback_qb_spot = np.array([center_pt[0] - 15.0, center_pt[1]]) 
                dists_to_qb = np.linalg.norm(centroids - fallback_qb_spot, axis=1)
                dists_to_qb[center_idx] = np.inf
                qb_idx = np.argmin(dists_to_qb)

            qb_pt = centroids[qb_idx]

            # Normalize coordinates relative to Center (Center = 0,0)
            normalized_pts = centroids - center_pt 
            
            remaining_pts = np.delete(normalized_pts, [center_idx, qb_idx], axis=0)
            remaining_pts = remaining_pts[remaining_pts[:, 1].argsort()]

            final_vector = np.vstack(([normalized_pts[center_idx]], [normalized_pts[qb_idx]], remaining_pts))

            cursor = self.db.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO formations_curated VALUES (?, ?, ?)", 
                           (clip_id, label, json.dumps(final_vector.tolist())))
            processed_count += 1

        self.db.conn.commit()
        print(f"[EXTRACTION] Successfully extracted canonical vectors for {processed_count} formations.")

class FormationML:
    """Handles Augmentation with Occlusion Simulation, Parquet export and XGBoost Training."""
    def __init__(self, db: TacticalDatabase):
        self.db = db

    def run_pipeline(self):
        df = pd.read_sql("SELECT * FROM formations_curated", self.db.conn)
        if df.empty:
            print("[ERROR] No curated formations found. Run Extractor first.")
            return

        dataset, labels = [], []
        SPATIAL_COPIES = 10
        OCCLUSION_COPIES = 15

        print("[ML] Generating synthetic dataset with spatial noise and simulated YOLO occlusions...")
        for _, row in df.iterrows():
            base_coords = np.array(json.loads(row['canonical_points']))
            label = row['label']

            # Original clean vector
            dataset.append(base_coords.flatten())
            labels.append(label)

            # 1. Spatial Noise Augmentation (All 11 players visible)
            for _ in range(SPATIAL_COPIES):
                aug_coords = base_coords.copy()
                aug_coords += np.random.normal(0, 3.0, aug_coords.shape)
                aug_coords[:, 1] += np.random.uniform(-40, 40)
                dataset.append(aug_coords.flatten())
                labels.append(label)

            # 2. Synthetic Occlusion Augmentation (Simulating missing YOLO detections via NaN)
            # We randomly drop between 1 and 4 non-Center/non-QB players
            for _ in range(OCCLUSION_COPIES):
                aug_coords = base_coords.copy()
                aug_coords += np.random.normal(0, 3.5, aug_coords.shape)
                aug_coords[:, 1] += np.random.uniform(-40, 40)
                
                # Indices 2 to 10 correspond to P1..P9 (leave Center [0] and QB [1] intact)
                num_dropped = np.random.randint(1, 5)
                dropped_indices = np.random.choice(range(2, 11), size=num_dropped, replace=False)
                aug_coords[dropped_indices] = np.nan
                
                dataset.append(aug_coords.flatten())
                labels.append(label)

        cols = ['C_x', 'C_y', 'QB_x', 'QB_y']
        for i in range(1, 10): cols.extend([f'P{i}_x', f'P{i}_y'])
        
        final_df = pd.DataFrame(dataset, columns=cols)
        final_df['label'] = labels
        final_df.to_parquet(DATASET_EXPORT_PATH)
        print(f"[ML] Augmented dataset saved to {DATASET_EXPORT_PATH.name} ({len(final_df)} samples).")

        print("[ML] Training XGBoost Classifier with native NaN handling...")
        X = final_df.drop('label', axis=1)
        
        unique_labels = final_df['label'].unique()
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = final_df['label'].map(label_map)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = xgb.XGBClassifier(
            objective='multi:softprob', 
            num_class=len(unique_labels),
            max_depth=5, 
            learning_rate=0.08,
            n_estimators=300,
            eval_metric='mlogloss',
            missing=np.nan  # Explicitly instruct XGBoost to route NaN splits optimally
        )
        
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        accuracy = clf.score(X_test, y_test)
        
        print(f"\n🚀 [SUCCESS] XGBoost Model Accuracy (with simulated occlusions): {accuracy*100:.2f}%")
        clf.save_model(BASE_DIR / "formation_xgboost_v1.json")
        
        with open(BASE_DIR / "formation_labels_map.json", "w") as f:
            json.dump({v: k for k, v in label_map.items()}, f)
        print("[ML] Model and label map exported successfully.")

if __name__ == "__main__":
    print("=== TACTICAL FORMATION ARCHITECT ===")
    db = TacticalDatabase()
    
    print("\nPhase 1: Extracting canonical formations from historical tracking data...")
    extractor = FormationExtractor(db)
    extractor.extract_all_formations()
    
    print("\nPhase 2: Training XGBoost with Synthetic Occlusions...")
    ml = FormationML(db)
    ml.run_pipeline()