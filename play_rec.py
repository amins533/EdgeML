import sqlite3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "plays.db"
CSV_BREAKDOWN_PATH = BASE_DIR / "breakdown-2024.csv" 

def parse_clip_index(clip_id_str):
    if pd.isna(clip_id_str): return None
    digits = ''.join(filter(str.isdigit, str(clip_id_str)))
    return int(digits) if digits else None

def get_breakdown_mapping():
    if not CSV_BREAKDOWN_PATH.exists(): return {}
    try:
        with open(CSV_BREAKDOWN_PATH, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
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
        if resolved_category:
            mapping[clip_num] = {'label': resolved_category}
    return mapping

def build_dense_kinematic_dataset():
    excel_map = get_breakdown_mapping()
    if not DB_PATH.exists(): return np.array([]), np.array([])
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT clip_id, post_snap_matrix FROM play_trajectories")
    rows = cursor.fetchall()
    conn.close()

    features_list = []
    labels_list = []
    checkpoints = [15, 35, 55, 75]

    for clip_id, matrix_json in rows:
        clip_num = parse_clip_index(clip_id)
        if clip_num is None or clip_num not in excel_map: continue
        play_concept_label = excel_map[clip_num]['label']

        raw_matrix = json.loads(matrix_json)
        max_frames = 90
        trackers_data = {}
        
        for t_id, data in raw_matrix.items():
            timeline = data["timeline"] if isinstance(data, dict) else data
            trackers_data[t_id] = timeline[:max_frames]

        initial_frame = None
        for f in range(25):
            valid_pts = [trackers_data[t_id][f] for t_id in trackers_data if f < len(trackers_data[t_id]) and trackers_data[t_id][f] is not None and trackers_data[t_id][f][0] is not None]
            if len(valid_pts) >= 5:
                initial_frame = f
                break
        if initial_frame is None: continue

        snap_pts = [trackers_data[t_id][initial_frame] for t_id in trackers_data if initial_frame < len(trackers_data[t_id]) and trackers_data[t_id][initial_frame] is not None and trackers_data[t_id][initial_frame][0] is not None]
        snap_x = [p[0] for p in snap_pts]
        snap_y = [p[1] for p in snap_pts]
        
        center_x_0 = np.median(snap_x)
        center_y_0 = np.median(snap_y)
        scale_x = np.std(snap_x) if np.std(snap_x) > 0 else 1.0
        scale_y = np.std(snap_y) if np.std(snap_y) > 0 else 1.0

        core_trackers = []
        perimeter_trackers = []
        for t_id, timeline in trackers_data.items():
            if initial_frame < len(timeline) and timeline[initial_frame] is not None and timeline[initial_frame][0] is not None:
                p0 = timeline[initial_frame]
                dist = np.sqrt(((p0[0] - center_x_0)/scale_x)**2 + ((p0[1] - center_y_0)/scale_y)**2)
                if dist < 1.2:
                    core_trackers.append(t_id)
                else:
                    perimeter_trackers.append(t_id)

        play_vector = []

        for cp in checkpoints:
            for group, g_trackers in [("core", core_trackers), ("perimeter", perimeter_trackers)]:
                group_pts = []
                for t_id in g_trackers:
                    if cp < len(trackers_data[t_id]) and trackers_data[t_id][cp] is not None and trackers_data[t_id][cp][0] is not None:
                        group_pts.append(trackers_data[t_id][cp])
                
                if not group_pts:
                    play_vector.extend([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
                    continue
                
                gx = [p[0] for p in group_pts]
                gy = [p[1] for p in group_pts]
                
                # Clean, normalized continuous coordinate positions
                curr_shift_x = (np.median(gx) - center_x_0) / scale_x
                curr_shift_y = (np.median(gy) - center_y_0) / scale_y
                curr_spread_x = np.std(gx) / scale_x
                curr_spread_y = np.std(gy) / scale_y
                
                # Maximum travel displacement inside this window frame
                max_displacement_x = (np.max(gx) - np.min(gx)) / scale_x
                max_displacement_y = (np.max(gy) - np.min(gy)) / scale_y
                
                play_vector.extend([curr_shift_x, curr_shift_y, curr_spread_x, curr_spread_y, max_displacement_x, max_displacement_y])

        features_list.append(play_vector)
        labels_list.append(play_concept_label)

    return np.array(features_list), np.array(labels_list)

def main():
    print("==================================================")
    print("     DENSE CONTINUOUS TRAJECTORY MODEL          ")
    print("==================================================")

    X_raw, y_mapped = build_dense_kinematic_dataset()
    if len(X_raw) == 0:
        print("[ERROR] Feature grid extraction pipeline produced empty matrices.")
        return

    print(f"[INFO] Profiled {len(X_raw)} plays down into a refined {len(X_raw[0])}-dimension dense matrix.")
    
    label_encoder = LabelEncoder()
    y_raw = label_encoder.fit_transform(y_mapped)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw if len(np.unique(y_raw)) > 1 else None
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([weight_dict[y] for y in y_train])

    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(label_encoder.classes_),
        max_depth=4,                     
        learning_rate=0.05,              
        n_estimators=400,               
        subsample=0.8,                  
        colsample_bytree=0.8,           
        eval_metric='mlogloss',
        early_stopping_rounds=30,       
        random_state=42
    )

    print(f"[TRAINING] Optimizing dense gradient booster... Test size: {len(X_test)}")
    clf.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_test, y_test)], verbose=False)

    predictions = clf.predict(X_test)
    
    print("\n" + "="*50)
    print("     DENSE KINEMATIC PERFORMANCE REPORT     ")
    print("="*50)
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))
    
    print("="*50)
    print("                  CONFUSION MATRIX               ")
    print("="*50)
    cm = confusion_matrix(y_test, predictions)
    header = f"{'True \\ Pred':<15}" + "".join([f"{c[:10]:>12}" for c in label_encoder.classes_])
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(cm):
        row_str = f"{label_encoder.classes_[idx]:<15}" + "".join([f"{val:>12}" for val in row])
        print(row_str)

    joblib.dump(clf, "play_classifier.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("\n Saved updated dense feature models to disk.")

    final_acc = accuracy_score(y_test, predictions)
    print("\n" + "═"*58)
    print(f"   [SUMMARY] REFINED KINEMATIC ACCURACY: {final_acc * 100:.2f}% ")
    print("═"*58 + "\n")

if __name__ == "__main__":
    main()