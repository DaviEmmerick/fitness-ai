import cv2
import csv
import os
from pathlib import Path
from ultralytics import YOLO

INPUT_DIR = Path("knn_dataset/Segmented Dataset") 
OUTPUT_CSV = "dataset_treino_knn.csv"
MODEL_PATH = "../best.pt"

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

def extract_features_recursive():
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo {MODEL_PATH} não encontrado.")
        return
    
    print(f"Carregando YOLO...")
    model = YOLO(MODEL_PATH)

    if not INPUT_DIR.exists():
        print(f"Pasta de entrada não encontrada: {INPUT_DIR}")
        return

    header = []
    for i in range(17):
        header.append(f"x{i}")
        header.append(f"y{i}")
    header.append("label") 
    header.append("exercicio")

    print(f"Varrendo vídeos recursivamente em: {INPUT_DIR}")
    
    total_frames = 0
    videos_processados = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for exercise_folder in INPUT_DIR.iterdir():
            if exercise_folder.is_dir():
                print(f"\nEntrando em: {exercise_folder.name}")
                

                todos_videos = []
                for ext in VIDEO_EXTENSIONS:
                    todos_videos.extend(list(exercise_folder.rglob(f"*{ext}")))
                
                print(f"Encontrados {len(todos_videos)} vídeos submersos.")

                for video_path in todos_videos:
                    nome_arquivo = video_path.name.lower()
                    
                    if nome_arquivo.startswith("g") or "_g_" in nome_arquivo:
                        label = 1 
                    elif nome_arquivo.startswith("b") or "_b_" in nome_arquivo:
                        label = 0 
                    else:
                        continue 

                    cap = cv2.VideoCapture(str(video_path))
                    frame_count_local = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        results = model.predict(frame, verbose=False, conf=0.5)
                        
                        if results[0].keypoints is not None:
                            kpts = results[0].keypoints.xyn.cpu().numpy()[0]
                            
                            if len(kpts) == 17:
                                linha = kpts.flatten().tolist()
                                linha.append(label)
                                linha.append(exercise_folder.name)
                                writer.writerow(linha)
                                
                                frame_count_local += 1
                                total_frames += 1
                    
                    cap.release()
                    videos_processados += 1
                    
                    if videos_processados % 10 == 0:
                        print(f"processados {videos_processados} vídeos até agora.")

    print(f"\nSUCESSO FINAL!")
    print(f"Total de vídeos lidos: {videos_processados}")
    print(f"Total de poses salvas: {total_frames}")
    print(f"Arquivo CSV gerado: {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features_recursive()