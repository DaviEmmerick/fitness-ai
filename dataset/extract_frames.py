import cv2
import os
from pathlib import Path

INPUT_DIR = Path("dataset/verified_data/verified_data/data_crawl_10s")
OUTPUT_DIR = Path("train_data/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FRAMES_PER_SECOND = 2  
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def extract_frames():
    if not INPUT_DIR.exists():
        print(f"ERRO CRÍTICO: O caminho não foi encontrado: {INPUT_DIR.resolve()}")
        print("Dica: Verifique se você está rodando o script da RAIZ do projeto.")
        return

    print(f"Lendo vídeos de: {INPUT_DIR}")
    print(f"Salvando frames em: {OUTPUT_DIR}\n")

    total_videos = 0
    total_frames = 0

    for exercise_folder in INPUT_DIR.iterdir():
        if exercise_folder.is_dir():
            
            videos = [v for v in exercise_folder.iterdir() 
                      if v.suffix.lower() in VIDEO_EXTENSIONS]
            
            if not videos:
                continue
                
            print(f"{exercise_folder.name}: Encontrados {len(videos)} vídeos.")
            
            for video_path in videos:
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                if fps <= 0: 
                    continue
                
                hop = round(fps / FRAMES_PER_SECOND)
                if hop < 1: hop = 1
                
                frame_count = 0
                saved_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_count % hop == 0:
                        frame_name = f"{exercise_folder.name}_{video_path.stem}_{saved_count}.jpg"
                        save_path = OUTPUT_DIR / frame_name
                        
                        cv2.imwrite(str(save_path), frame)
                        saved_count += 1
                        total_frames += 1
                    
                    frame_count += 1
                cap.release()
                total_videos += 1

    print(f"\nCONCLUÍDO!")
    print(f"Total de vídeos processados: {total_videos}")
    print(f"Total de frames extraídos: {total_frames}")

if __name__ == "__main__":
    extract_frames()