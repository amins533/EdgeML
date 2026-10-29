import os
import random
import numpy as np
import cv2
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
TRAIN_DIR = Path("sortedcards/train")

def augment_image(img):
    """
    Aplica transformaciones seguras para tarjetas de juego tácticas.
    Evita deformar drásticamente para mantener el sentido de la jugada.
    """
    h, w = img.shape[:2]
    
    # 1. Desplazamiento (Shift) muy leve: max ~1.5% (equivalente a ~media yarda)
    # Esto cumple con tu restricción estricta de no desplazar a los jugadores de sus gaps
    tx = random.uniform(-0.015, 0.015) * w
    ty = random.uniform(-0.015, 0.015) * h
    
    # 2. Rotación levísima (-2 a +2 grados)
    angle = random.uniform(-2, 2)
    
    # 3. Escalado leve (98% a 102%)
    scale = random.uniform(0.98, 1.02)
    
    # Matriz de transformación Afín
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Aplicar transformación geométrica con fondo replicado
    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 4. Ruido Gaussiano (Pixel-level) muy sutil
    # Agrega textura visual sin mover coordenadas. Std dev bajita controlada.
    if random.random() > 0.3:  # 70% de probabilidad de aplicar ruido
        noise = np.random.normal(0, random.uniform(2, 5), aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return aug_img

def main():
    print("==================================================")
    print("⚖️ AUMENTANDO Y BALANCEANDO EL DATASET DE ENTRENAMIENTO")
    print("==================================================")
    
    if not TRAIN_DIR.exists():
        print(f"❌ [ERROR] No se encontró la carpeta: {TRAIN_DIR}")
        return
        
    # 1. Analizar el tamaño actual de cada subcarpeta en 'train'
    classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    class_counts = {}
    
    for cls in classes:
        images = list(cls.glob("*.png"))
        class_counts[cls.name] = len(images)
        
    if not class_counts:
        print("❌ No hay clases para aumentar en train.")
        return
        
    # 2. Calcular el Target de Balanceo
    max_class_size = max(class_counts.values())
    print(f"📊 La clase mayoritaria ('{max(class_counts, key=class_counts.get)}') tiene {max_class_size} imágenes.")
    
    # El target será por lo menos 200, o el doble de la clase mayoritaria (el que sea más grande).
    # Así nos aseguramos de que TODAS las clases aumenten significativamente de tamaño.
    global_target = max(200, max_class_size * 2)
    print(f"🎯 Target de Balanceo fijado en: {global_target} imágenes EXACTAS por clase.")
    print("--------------------------------------------------")
    
    total_augmented = 0
    
    # 3. Aplicar Data Augmentation para igualar todas las clases al target
    for cls in classes:
        images = list(cls.glob("*.png"))
        current_count = len(images)
        
        if current_count == 0:
            continue
            
        needed = global_target - current_count
        
        if needed > 0:
            print(f"⚙️ Aumentando '{cls.name}': {current_count} -> {global_target} (+{needed} nuevas)")
            
            images_list = [str(p) for p in images]
            
            for i in range(needed):
                # Tomar aleatoriamente una imagen base existente de esta clase
                src_img_path = random.choice(images_list)
                img = cv2.imread(src_img_path)
                
                if img is None:
                    continue
                    
                # Aplicar la pipeline segura de aumentación
                aug_img = augment_image(img)
                
                # Guardar con un nuevo nombre
                new_filename = f"aug_{i}_{Path(src_img_path).name}"
                new_filepath = cls / new_filename
                cv2.imwrite(str(new_filepath), aug_img)
                
                total_augmented += 1
        else:
            print(f"✅ '{cls.name}' ya tiene suficientes imágenes ({current_count}).")
            
    print("==================================================")
    print(f"🎉 ¡ÉXITO TOTAL! Se generaron {total_augmented} imágenes sintéticas limpias.")
    print(f"📁 Tu carpeta '{TRAIN_DIR}' ahora está DRÁSTICAMENTE BALANCEADA.")

if __name__ == "__main__":
    main()
