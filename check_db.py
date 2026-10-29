import sqlite3
import json

conn = sqlite3.connect("plays.db")
cursor = conn.cursor()
cursor.execute("SELECT clip_id, post_snap_matrix FROM play_trajectories LIMIT 1")
row = cursor.fetchone()
conn.close()

if row:
    clip_id, matrix_json = row
    print(f"--- Diagnosing Clip: {clip_id} ---")
    data = json.loads(matrix_json)
    
    # Get the first tracked player
    first_track_id = list(data.keys())[0]
    timeline = data[first_track_id]["timeline"] if isinstance(data[first_track_id], dict) else data[first_track_id]
    
    print(f"Total frames stored: {len(timeline)}")
    print(f"Frame 0 (Snap) Coords : {timeline[0]}")
    print(f"Frame 10 Coords        : {timeline[min(10, len(timeline)-1)]}")
    print(f"Frame 22 Coords        : {timeline[min(22, len(timeline)-1)]}")
    print(f"Frame 35 Coords        : {timeline[min(35, len(timeline)-1)]}")
    print(f"Frame 50 Coords        : {timeline[min(50, len(timeline)-1)]}")
else:
    print("Database is empty.")