# EdgeML Meets American Football

**Modul:** Informatik Projekt im 5. Semester  
**Institution:** Frankfurt University of Applied Sciences  
**Team:** Gruppe 1  
**Betreuer:** David Merkl, M.Sc. (Labor für eingebettete Systeme)  
**Datum:** August 2026  

Das vorliegende Repository enthält ein vollständiges Edge AI System für die taktische Analyse von American Football Spielen. Dieses Universitätsprojekt wurde nach den Prinzipien des agilen Scrum Frameworks entwickelt. Das Projekt kombiniert Computer Vision und Machine Learning, um Spielzüge, Formationen und Laufwege in Echtzeit zu erkennen. 

Die Serverversion des Systems wurde ursprünglich für den eigenständigen Einsatz auf Edge Hardware wie dem Nvidia Jetson Orin Nano am Spielfeldrand konzipiert. Diese Version lässt sich jedoch ebenso problemlos auf anderen leistungsstarken Serverumgebungen betreiben.

---

## Systemübersicht & Pipeline

Die Analysepipeline verarbeitet Football-Videos asynchron durch die folgenden sequenziellen Phasen:

1. Frame-Verarbeitung und asynchrones Video-Streaming
2. Spielererkennung und Multi-Objekt-Tracking
3. Homographie-Schätzung (Perspektiven-Transformation)
4. Trikotnummern-Erkennung (OCR)
5. Formationsklassifizierung (Pre-Snap)
6. Datenbankspeicherung und Trajektorien-Aufzeichnung
7. Taktische Playcard-Generierung

Das Ziel ist es, taktische Informationen aus aufgezeichneten Football-Spielen vollautomatisch zu extrahieren.

---

## Projektstruktur

Das System ist in verschiedene logische Schichten und Ordner unterteilt, um den Code sauber zu trennen:

```text
models/                 Trainierte Gewichte für die YOLO- und XGBoost-Modelle
videos/                 Eingabevideos (muss lokal erstellt werden)
project-web-hub/        Quellcode für React-Frontend und FastAPI-Backend
dataset_preparation/    Skripte zur Datenaufbereitung und Konvertierung
training/               Skripte für das Training der KI-Modelle
tools/                  Diverse kleine Skripte für Tests und Debugging
cards/                  Generierte, finale Playcards nach der Videoanalyse
homographie_analisis/   Debugging und Fehleranalysen der Feldkoordinaten
ocr_real_runs/          Lokal generierte Bilder und Evaluierungen des OCR-Modells
data-base/              Speicherort der generierten SQLite-Datenbanken
Test Versionen/         Experimentelle Implementierungen
```

---

## Systemkomponenten & Hauptdateien

Die Kernfunktionalität wird durch folgende essenzielle Skripte abgebildet:

* **`main.py`** – Der primäre Orchestrator für die rein lokale Ausführung des Programms mit direkter Bildschirmausgabe.
* **`main-webhub-integration.py`** – Eine modifizierte Version für den Serverbetrieb, welche die Inferenz in einen separaten Thread auslagert und mit dem Backend kommuniziert.
* **`video_pipeline.py`** – Liest Video-Frames asynchron über eine Producer-Consumer-Architektur, um I/O-Blockaden zu vermeiden.
* **`homography_engine.py`** – Der Motor für die Transformation der Spielerpositionen von Bildkoordinaten in ein zweidimensionales Taktikfeld.
* **`ocr_pipeline.py`** – Die zentrale Datenpipeline für die Evaluierung der Lesbarkeit und die asynchrone Erkennung der Trikotnummern mittels EasyOCR.
* **`heatmap.py`** – Generiert aus den getrackten Spieler-Trajektorien die taktischen Playcards.
* **`data_base.py` & `data_class.py`** – Verwalten die Datenstrukturen und das Speichern der Ergebnisse in SQLite-Datenbanken (läuft in einem Hintergrund-Thread).
* **`formation_architect.py`** – Extrahiert relevante Formationen aus der Datenbank und trainiert das XGBoost Modell.

---

## Installation und Vorbereitung

> **Hinweis zur Python Version:** Für eine reibungslose Ausführung wird strikt **Python 3.12.3** empfohlen. Neuere Versionen (wie 3.13) können ebenfalls genutzt werden, erfordern jedoch möglicherweise eine manuelle Anpassung und Konfliktauflösung bei den Dependencies.

Für eine saubere Ausführung wird dringend die Einrichtung einer isolierten virtuellen Umgebung empfohlen. Dadurch werden Konflikte mit bestehenden Systempaketen vermieden:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Wichtige Hinweise zur Dateistruktur:**
Die Versionskontrolle schließt bewusst bestimmte Verzeichnisse und Dateitypen aus. Erstellen Sie nach dem Klonen des Repositories manuell einen Ordner namens `videos` im Wurzelverzeichnis des Projektes und fügen Sie dort die zu analysierenden Videodateien ein. 

Zusätzlich ignoriert Git automatisch generierte Ordner wie `data-base`, `datasets`, `runs`, `videos-ocr` sowie `ocr_real_runs`, um Speicherplatz zu sparen. Gleiches gilt für lokale SQLite Datenbanken und grundlegende YOLO Basisgewichte. Diese Maßnahme hält das Repository kompakt.

**Hinweis für Mac OS Nutzer:**
Nutzer von macOS mit Apple Silicon Chips müssen bei der Installation der Abhängigkeiten besonders aufmerksam sein. Pakete wie PyTorch und OpenCV erfordern spezifische Versionen für die ARM Architektur, da das System MPS für die Hardwarebeschleunigung nutzt. 
*Zudem wird der Betrieb des Webservers auf einem Mac System derzeit nicht empfohlen.* Die aktuelle Version weist eine ungelöste Inkompatibilität bei der Videoübertragung über HTTP auf. Dieses Problem resultiert aus massiven Konflikten zwischen PyTorch und OpenCV, wenn diese in sekundären Threads unter macOS ausgeführt werden.

---

## Lokale Ausführung

Für die Analyse direkt auf dem Rechner bietet das Hauptskript verschiedene Modi. Der Standardmodus öffnet ein Fenster zur visuellen Verfolgung der Echtzeitanalyse:

```bash
python3 main.py
```

Für eine ressourcenschonende Ausführung im Hintergrund existiert ein spezieller Parameter. Dieser Modus deaktiviert die grafische Oberfläche und eignet sich perfekt, wenn ausschließlich die Generierung der Playcards gewünscht wird oder kein Bildschirm zur Verfügung steht:

```bash
python3 main.py --headless
```

---

## Web Hub Ausführung

Damit externe Endgeräte (z.B. Smartphones der Trainer) erfolgreich mit dem Backend kommunizieren können, muss vor dem Start eine wichtige Netzwerkkonfiguration angepasst werden. 

Öffnen Sie die Datei `project-web-hub/frontend/src/services/api.js` in einem Texteditor. Ersetzen Sie dort den standardmäßigen Wert `localhost` durch die tatsächliche IP-Adresse, welche das Jetson/Edge-Gerät beim Erstellen des Hotspots zugewiesen bekommt. Ohne diese Änderung suchen verbundene Endgeräte das Backend fälschlicherweise lokal auf sich selbst.

Für den Einsatz am Spielfeldrand erfordert das System die parallele Ausführung von Backend und Frontend in zwei separaten Terminals:

**1. FastAPI Server (Terminal 1):**
```bash
cd project-web-hub/backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**2. React Frontend (Terminal 2):**
```bash
cd project-web-hub/frontend
pnpm install
pnpm run dev -- --host
```

Trainer können anschließend über den Browser ihrer mobilen Endgeräte auf das Dashboard zugreifen und die Echtzeitanalyse steuern.

---

## Generierte Ausgaben (Output)

Am Ende der Verarbeitung generiert die Pipeline vollautomatisch:

- **Spieler-Detektionen und Multi-Object-Tracking**
- **Spieler-Trajektorien** (geglättete Laufwege)
- **Erkannte Trikotnummern** (Re-ID durch OCR)
- **Offensiv-Formationsvorhersagen** (Pre-Snap)
- **SQLite-Datenbanken** (enthalten alle Analyseresultate und Metadaten, gespeichert im Ordner `data-base/`)
- **Taktische Playcards** (visuelle Renderings der Spielzüge, gespeichert im Ordner `cards/`)
