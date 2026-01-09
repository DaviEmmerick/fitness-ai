import kagglehub
import shutil
import os

local_path = "dataset/"

if not os.path.exists(local_path):
    os.makedirs(local_path)
    print(f"Pasta '{local_path}' criada com sucesso.")

print("Baixando dataset...")
cache_path = kagglehub.dataset_download("philosopher0808/gym-workoutexercises-video")

print(f"Movendo arquivos de {cache_path} para {local_path}...")

for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(local_path, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d) 
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Concluído! O dataset agora está visível na sua pasta: {os.path.abspath(local_path)}")