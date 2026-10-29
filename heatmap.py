import os
import shutil
import random
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

# --- SYSTEM DIRECTORY SETUP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "play_recognition" / "best.pt"

def generate_numbered_yard_playcard(clip_name="play_card", db_path="tactical_system.db", formation_text="Unknown"):
    print(f"\n Generating vector playcard for clip: '{clip_name}'...")
    
    if not os.path.exists(db_path) and os.path.exists("football_system.db"):
        db_path = "football_system.db"
        
    if not os.path.exists(db_path):
        print(f" Database path {db_path} does not exist yet. Skipping playcard.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        table_name = "offense_trajectory_matrix" if "offense_trajectory_matrix" in tables else (tables[0] if tables else None)
        
        if not table_name:
            print(" No tracking tables found in database.")
            conn.close()
            return
            
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f" Error locating database tracking matrix: {e}")
        conn.close()
        return

    if df.empty:
        print(" Tracking matrix is empty. Skipping playcard generation.")
        conn.close()
        return

    # Extract jersey mapping safely
    jersey_map = {}
    if "players" in tables:
        try:
            df_players = pd.read_sql_query("SELECT tracking_id, jersey_number FROM players", conn)
            for _, row_p in df_players.iterrows():
                if pd.isna(row_p['tracking_id']) or row_p['tracking_id'] == '-': 
                    continue
                t_id = int(float(row_p['tracking_id']))
                j_num = row_p['jersey_number']
                if j_num and str(j_num).strip() and str(j_num) != 'None' and str(j_num) != '-':
                    jersey_map[t_id] = str(int(float(j_num)))
        except Exception:
            pass

    frame_cols = [c for c in df.columns if c.startswith('Frame_')]
    
    def parse_xy(val):
        if val == '-' or pd.isna(val): return None
        try: return ast.literal_eval(val)
        except: return None

    profiles = {}
    all_x, all_y = [], []
    
    for idx, row in df.iterrows():
        try:
            p_id = int(row.get('YOLO_Tracking_ID', row.iloc[0]))
        except:
            continue
        role = row.get('Role', 'Player')
        jersey_num = row.get('Jersey_Number', '-') if 'Jersey_Number' in df.columns else '-'
        
        coords = {}
        for c in frame_cols:
            f_num = int(c.split('_')[1])
            parsed = parse_xy(row[c])
            if parsed: 
                coords[f_num] = parsed
                all_x.append(parsed[0])
                all_y.append(parsed[1])
            
        if coords:
            profiles[p_id] = {'coords': coords, 'role': role, 'jersey': jersey_num}

    if not all_x or not all_y:
        print(f" No tracking matrix coordinates discovered for {clip_name}.")
        conn.close()
        return

    top_11_keys = sorted(profiles.keys(), key=lambda k: len(profiles[k]['coords']), reverse=True)
    if len(top_11_keys) > 11:
        top_11_keys = top_11_keys[:11]

    #FIELD VISUAL DESIGN SYSTEM
    fig, ax = plt.subplots(figsize=(24, 13.5))
    bg_color = '#226b39'       
    border_color = '#11381d'   
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(border_color)

    min_field_x, max_field_x = min(all_x), max(all_x)
    min_field_y, max_field_y = min(all_y), max(all_y)
    
    span_x = max(max_field_x - min_field_x, 10.0)
    span_y = max(max_field_y - min_field_y, 10.0)
    
    # 1. VERTICAL YARD LINES
    x_lines = np.linspace(min_field_x, max_field_x, 7)
    yard_numbers = [10, 20, 30, 40, 50, 40, 30] 
    
    for idx, x_pos in enumerate(x_lines):
        ax.axvline(x=x_pos, color='white', linestyle='-', linewidth=2, alpha=0.3, zorder=1)
        num_str = str(yard_numbers[idx % len(yard_numbers)])
        arrow_indicator = "◀ " if idx < 4 else " ▶"
        display_text = f"{num_str}" if idx == 4 else (f"{arrow_indicator}{num_str}" if idx < 4 else f"{num_str}{arrow_indicator}")
        
        ax.text(x_pos, min_field_y - 15, display_text, color='white', alpha=0.4, 
                fontsize=14, fontweight='bold', ha='center', va='center', rotation=180) 
        ax.text(x_pos, max_field_y + 15, display_text, color='white', alpha=0.4, 
                fontsize=14, fontweight='bold', ha='center', va='center')

    # 2. VERTICAL LINE OF SCRIMMAGE
    los_x = None
    for p_id in top_11_keys:
        if 'center' in str(profiles[p_id]['role']).lower():
            start_f = min(profiles[p_id]['coords'].keys())
            los_x = profiles[p_id]['coords'][start_f][0]
            break
            
    if los_x is None and top_11_keys:
        try:
            initial_xs = [profiles[k]['coords'][min(profiles[k]['coords'].keys())][0] for k in top_11_keys]
            los_x = np.median(initial_xs)
        except:
            los_x = min_field_x

    if los_x is not None:
        ax.axvline(x=los_x, color='#00a2ff', linestyle='-', linewidth=5, alpha=0.9, label='Line of Scrimmage', zorder=2)

    # 2.5 ANOMALOUS ALIGNMENT FILTER (LOS GUARDRAIL)
    # Find out which direction the offense is moving from.
    # If the majority of players start behind los_x, then offense is on the LEFT side.
    initial_offsets = []
    for k in top_11_keys:
        start_f = min(profiles[k]['coords'].keys())
        initial_offsets.append(profiles[k]['coords'][start_f][0] - los_x)
        
    # True if offense is on the left side of the screen, False if on the right side
    offense_is_left = np.median(initial_offsets) < 0

    # Max allowed alignment cushion past the line of scrimmage before the play (in tracking units)
    alignment_buffer = 1.5 

    # 3. PLAYER TRAJECTORY PATHS
    color_palette = ['#ffffff', '#00ffd0', '#ff4a4a', '#ffcc00', '#70a1ff', '#eccc68', '#ff6b81', '#7bed9f', '#ffa502', '#a4b0be']

    for i, master_id in enumerate(top_11_keys):
        p = profiles[master_id]
        role = p['role']
        sorted_frames = sorted(p['coords'].keys())
        
        x_raw = [p['coords'][f][0] for f in sorted_frames]
        y_raw = [p['coords'][f][1] for f in sorted_frames]
        
        #Apply the LOS filter to early/pre-snap frames
        # Adjust the first 3 frames if they are found across the line illegally
        for frame_idx in range(min(3, len(x_raw))):
            if offense_is_left:
                if x_raw[frame_idx] > (los_x + alignment_buffer):
                    x_raw[frame_idx] = los_x - alignment_buffer # Clamp them back safely behind the LOS
            else:
                if x_raw[frame_idx] < (los_x - alignment_buffer):
                    x_raw[frame_idx] = los_x + alignment_buffer

        if len(x_raw) > 5:
            try:
                x_smooth = gaussian_filter1d(x_raw, sigma=1.5)
                y_smooth = gaussian_filter1d(y_raw, sigma=1.5)
            except:
                x_smooth, y_smooth = x_raw, y_raw
        else:
            x_smooth, y_smooth = x_raw, y_raw

        color = color_palette[i % len(color_palette)]
        
        ax.plot(x_smooth, y_smooth, color=color, linewidth=4.5, alpha=0.9, zorder=4)
        ax.scatter(x_smooth[0], y_smooth[0], color=color, s=200, facecolors='none', linewidths=3.5, zorder=5)
        
        if len(x_smooth) > 1:
            dx, dy = x_smooth[-1] - x_smooth[-2], y_smooth[-1] - y_smooth[-2]
            dist = np.hypot(dx, dy)
            if dist > 0:
                ax.annotate('', xy=(x_smooth[-1], y_smooth[-1]), xytext=(x_smooth[-1] - (dx/dist)*10, y_smooth[-1] - (dy/dist)*10),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=4.5, mutation_scale=16, facecolor=color), zorder=5)
                
        jersey_val = p.get('jersey', '-')
        if (jersey_val == "-" or jersey_val is None) and master_id in jersey_map:
            jersey_val = jersey_map[master_id]

        if jersey_val != "-" and jersey_val is not None and str(jersey_val).strip() != '':
            label_text = f"ID {master_id} (# {jersey_val}) ({role[:4]})"
        else:
            label_text = f"ID {master_id} ({role[:4]})"
            
        ax.text(x_smooth[-1] + 12, y_smooth[-1], label_text, color='white', fontsize=10, fontweight='bold', va='center')

    # 4. SCALE PROCESSING BOUNDS
    ax.set_xlim(min_field_x - (span_x * 0.15), max_field_x + (span_x * 0.15))
    ax.set_ylim(min_field_y - (span_y * 0.15), max_field_y + (span_y * 0.15))
    
    ax.invert_yaxis() 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values(): 
        spine.set_visible(False)

    plt.tight_layout()
    
    # Render temporary flat image for YOLO evaluation
    os.makedirs("cards", exist_ok=True)
    temp_output_path = os.path.join("cards", f"{clip_name}_temp.png")
    plt.savefig(temp_output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)

    # --- IN-LINE INFERENCE ---
    recognized_play_label = "Model Unresponsive"
    if MODEL_PATH.exists():
        try:
            classifier = YOLO(str(MODEL_PATH))
            inference_results = classifier.predict(source=temp_output_path, imgsz=416, verbose=False)
            
            if inference_results and len(inference_results) > 0:
                res = inference_results[0]
                top_class_idx = res.probs.top1
                confidence_val = res.probs.top1conf.item() * 100
                label_string = res.names[top_class_idx]
                
                recognized_play_label = f"{label_string.upper()} ({confidence_val:.1f}%)"
        except Exception as err:
            recognized_play_label = f"Processing Error ({str(err)[:20]})"
    else:
        recognized_play_label = "best.pt Missing from /models/play_recognition/"

    # Remove temporary snapshot file safely
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    # 5. RENDER HUD HEADERS ONTO THE PLAYCARD
    ax.set_title(f"Tactical Playcard - {clip_name}", color='white', fontsize=22, pad=25, fontweight='bold')
    ax.text(0.02, 0.95, f"Formation: {formation_text}", 
            color='#00a2ff', fontsize=18, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='#11381d', alpha=0.8, edgecolor='#00a2ff', boxstyle='round,pad=0.4', lw=2))
            
    ax.text(0.02, 0.87, f"Recognized Play: {recognized_play_label}", 
            color='#ffcc00', fontsize=18, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='#11381d', alpha=0.8, edgecolor='#ffcc00', boxstyle='round,pad=0.4', lw=2))

    # Save finalized tactical playcard file
    output_path = os.path.join("cards", f"{clip_name}_playcard.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close(fig) 
    print(f"💾 Rendered file archived successfully to: '{output_path}'")

    # --- RESET DATABASE AND FLUSH ---
    try:
        print(f" Flushing data matrix for table '{table_name}'...")
        cursor.execute(f"DELETE FROM {table_name};")
        if "players" in tables:
            cursor.execute("DELETE FROM players;")
        conn.commit()
    except Exception as e:
        print(f" Notice: Database flush skipped or locked: {e}")
    finally:
        conn.close()