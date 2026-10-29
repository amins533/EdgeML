import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "breakdown.xlsx"

print("=== 🏈 SPREADSHEET HEAD AUDIT ===")
if EXCEL_PATH.exists():
    df = pd.read_excel(EXCEL_PATH)
    print(f"Total Rows Found: {len(df)}")
    print(f"Columns Available: {list(df.columns)}")
    print("\n--- FIRST 15 ROWS OF CORE LOGIC DATA ---")

    # Print out row indices alongside potential key columns to find the offset
    preview_cols = [col for col in ['PLAY #', 'OFF PLAY', 'PLAY DIR', 'PLAY TYPE'] if col in df.columns]
    preview_df = df[preview_cols].head(15).copy()
    preview_df['Physical_Excel_Row'] = preview_df.index + 2
    print(preview_df.to_string(index=False))
else:
    print(f"Spreadsheet missing at {EXCEL_PATH}")