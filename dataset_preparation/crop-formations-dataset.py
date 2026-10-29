import os
import cv2
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    basis_verzeichnis = base_dir / "datasets" / "dataset-formation"
    ziel_verzeichnis = base_dir / "datasets" / "dataset-formation-cropped"
    
    gueltige_klassen = ['ACE', 'CANNON', 'DEUCE', 'LEFT', 'RIGHT', 'TREY', 'TRIPS']
    ordner_struktur = ['train', 'val']
    
    gesamtzahl = 0

    print("Starte den intelligenten Bildzuschnitt...")

    for split in ordner_struktur:
        for klasse in gueltige_klassen:
            quellpfad = basis_verzeichnis / split / klasse
            zielpfad = ziel_verzeichnis / split / klasse
            
            zielpfad.mkdir(parents=True, exist_ok=True)
            
            if not quellpfad.exists():
                continue

            for bild_datei in quellpfad.glob("*.jpg"):
                img = cv2.imread(str(bild_datei))
                if img is None:
                    continue
                
                hoehe, breite = img.shape[:2]
                
                y1 = int(hoehe * 0.30)  # Schneidet 30% von oben ab
                y2 = int(hoehe * 0.90)  # Schneidet 10% von unten ab
                
                x1 = 0                  
                x2 = breite             
                
                ausschnitt = img[y1:y2, x1:x2]
                
                cv2.imwrite(str(zielpfad / bild_datei.name), ausschnitt)
                gesamtzahl += 1
                
        print(f"[{split}] Zuschnitt abgeschlossen.")

    print(f"Fertig! {gesamtzahl} Bilder wurden zugeschnitten.")

if __name__ == "__main__":
    main()