from ultralytics import YOLO
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "train_data" / "images"
LABELS_DIR = BASE_DIR / "train_data" / "labels"

LABELS_DIR.mkdir(parents=True, exist_ok=True)

def create_labels():
    print("Carregando modelo YOLOv11n-pose...")
    model = YOLO('yolo11n-pose.pt')

    print(f"Lendo imagens de: {IMAGES_DIR}")
    print(f"Salvando labels em: {LABELS_DIR}")

    images = list(IMAGES_DIR.glob("*.jpg"))
    
    if not images:
        print("Nenhuma imagem encontrada! Verifique se rodou o extract_frames.py")
        return

    results = model.predict(source=IMAGES_DIR, stream=True, conf=0.5, verbose=False)

    count = 0
    for result in results:
        image_name = Path(result.path).stem
        label_file = LABELS_DIR / f"{image_name}.txt"

        if len(result.boxes) == 0:
            continue

        with open(label_file, "w") as f:
            for i, box in enumerate(result.boxes):

                x, y, w, h = box.xywhn[0].tolist()
                
                kpts = result.keypoints.xyn[0].tolist()
                
                line = f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                
                for kp in kpts:
                    px, py = kp
                    visibility = 2 if px > 0 and py > 0 else 0
                    line += f" {px:.6f} {py:.6f} {visibility}"
                
                f.write(line + "\n")
        
        count += 1
        if count % 100 == 0:
            print(f"Processado: {count} imagens...")

    print(f"\nSucesso! {count} arquivos de label criados em {LABELS_DIR}")

if __name__ == "__main__":
    create_labels()