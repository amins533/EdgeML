import os
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "sortedcards"

def main():
    print("==================================================")
    print("🚀 INICIANDO ENTRENAMIENTO YOLO CLASSIFICATION")
    print("==================================================")

    # Validar que el dataset existe
    if not DATASET_DIR.exists():
        print(f"❌ [ERROR] No se encontró el dataset en: {DATASET_DIR}")
        return

    # 1. Cargar el modelo pre-entrenado
    # Empezamos con la versión Nano. Si el validation accuracy se estanca, 
    # cámbialo a "yolo11s-cls.pt" (Small).
    model_name = "yolo26n-cls.pt" 
    print(f"📦 Cargando modelo base: {model_name}")
    model = YOLO(model_name)

    # 2. Iniciar el entrenamiento
    print("🔥 Arrancando el motor de entrenamiento...")
    
    results = model.train(
        data=str(DATASET_DIR),  # Ultralytics lee automáticamente las subcarpetas train/val/test
        epochs=500,             # Tu máximo deseado
        patience=50,            # Si en 50 epochs no mejora el validation loss, se detiene para evitar overfitting
        imgsz=640,              # Aumentamos la resolución para no perder los detalles de las líneas (default era 224)
        batch=16,               # GPU-friendly. Súbelo a 32 si tu tarjeta de video tiene +8GB de VRAM
        device=0,               # Fuerza el uso de la GPU (cambia a 'cpu' si no tienes CUDA)
        project="playcard_models", # Carpeta principal donde guardará los resultados
        name="yolo26n_plays",   # Subcarpeta de esta corrida específica
        pretrained=True,        # Usa pesos preentrenados (Transfer Learning)
        optimizer="auto",       # YOLO elige AdamW o SGD según la arquitectura
        verbose=True
    )

    print("==================================================")
    print(f"🎉 [ÉXITO] Entrenamiento finalizado.")
    print(f"📂 Los mejores pesos (best.pt) y gráficas se guardaron en: ./playcard_models/yolo11n_416px/weights/")

if __name__ == "__main__":
    main()