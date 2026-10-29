import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. DEINE DATEN (Hier später deine echten Werte eintragen!)
data = {
    "Version": ["v1\n(YOLO11n)", "v2\n(YOLO11n)", "v3\n(YOLO11s)", "v4\n(YOLO11s Aug)", "v5\n(YOLO26s 1500)", "v6\n(YOLO26n Rebal)"],
    "mAP50": [0.45, 0.48, 0.62, 0.65, 0.78, 0.85],
    "Precision": [0.50, 0.52, 0.68, 0.71, 0.80, 0.88],
    "Recall": [0.42, 0.45, 0.58, 0.60, 0.75, 0.82]
}
df = pd.DataFrame(data)

# 2. DESIGN-SETUP (Modern & Clean)
# Wir nutzen ein klares Theme ohne störende Hintergrundfarbe
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(12, 7))

# Moderne Farbpalette (ähnlich wie bei professionellen Dashboards)
color_map = '#2563eb'   # Kräftiges Blau für die wichtigste Metrik
color_prec = '#f59e0b'  # Warmes Orange
color_rec = '#10b981'   # Smaragdgrün

# 3. LINIEN ZEICHNEN
# mAP50 (Hauptmetrik extra hervorgehoben)
ax.plot(df["Version"], df["mAP50"], marker='o', markersize=10, linewidth=3.5, 
        color=color_map, label="mAP@50", zorder=4, markeredgecolor='white', markeredgewidth=2)

# Leichter Schatten/Farbverlauf unter der mAP-Kurve für einen modernen Look
ax.fill_between(df["Version"], df["mAP50"], alpha=0.1, color=color_map, zorder=1)

# Precision & Recall (etwas dezenter)
ax.plot(df["Version"], df["Precision"], marker='s', markersize=8, linewidth=2.5, 
        color=color_prec, label="Precision", zorder=3, markeredgecolor='white', markeredgewidth=1.5, alpha=0.8)
ax.plot(df["Version"], df["Recall"], marker='^', markersize=8, linewidth=2.5, 
        color=color_rec, label="Recall", zorder=3, markeredgecolor='white', markeredgewidth=1.5, alpha=0.8)

# 4. WERTE DIREKT AN DIE PUNKTE SCHREIBEN (Nur für mAP50, damit es nicht überladen wirkt)
for i, value in enumerate(df["mAP50"]):
    ax.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color=color_map, zorder=5)

# 5. ACHSEN & GITTER ANPASSEN
# Gitterlinien nur horizontal und sehr dezent
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#d1d5db')
ax.xaxis.grid(False)

# Störende Rahmenlinien (Spines) entfernen
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#9ca3af')

# Y-Achse formatieren (0 bis 1.0, mit 10% Abstand nach oben für die Text-Labels)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.1, 0.2))

# Beschriftungen
ax.set_title("Entwicklung der Modell-Performance über alle Iterationen", 
             fontsize=18, fontweight='bold', pad=20, color='#1f2937', loc='left')
ax.set_ylabel("Score (0.0 – 1.0)", fontsize=13, fontweight='bold', color='#4b5563', labelpad=15)
ax.tick_params(axis='x', labelsize=11, colors='#4b5563', pad=10)
ax.tick_params(axis='y', labelsize=11, colors='#4b5563')

# 6. LEGENDE PERFEKTIONIEREN
# Oben rechts platzieren, ohne störenden Kasten
ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#e5e7eb', 
          fontsize=12, borderpad=1, title="Metriken", title_fontproperties={'weight':'bold'})

# 7. SPEICHERN & ANZEIGEN
plt.tight_layout()
# Hohe Auflösung (dpi=400) und transparenter Hintergrund für Word/LaTeX
plt.savefig("yolo_performance_premium.png", dpi=400, bbox_inches='tight', transparent=False)
print("✅ Premium-Diagramm gespeichert als 'yolo_performance_premium.png'")