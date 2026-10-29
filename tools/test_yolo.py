from ultralytics import YOLO
import cv2
from pathlib import Path

# 1. Cargar el modelo YOLO (la versión 'nano' es la más ligera, ideal para el Jetson luego)
base_dir = Path(__file__).resolve().parent.parent
model = YOLO(str(base_dir / "yolo26n.pt")) 

# 2. Apuntar al video que ya tienes
video_path = str(base_dir / "videos" / "wide-clip-008.mp4")
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        break

    # 3. YOLO hace su magia: detecta y dibuja las cajas
    # predict() solo busca cajas, no hace tracking todavía
    # 3. Magia MOT: YOLO detecta Y rastrea dándole un ID único a cada jugador
    resultados = model.track(
        frame, 
        persist=True,           # Fundamental: Le dice a YOLO que recuerde los IDs de los frames anteriores
        tracker="bytetrack.yaml", # Activamos el MOT-Framework ByteTrack
        classes=[0]             # Seguimos filtrando solo por humanos
    )
    
    # 4. YOLO tiene un método integrado para dibujar las cajas en la foto
    frame_anotado = resultados[0].plot()

    # 5. Mostrar el resultado
    cv2.imshow("YOLOv8 - Prueba", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()