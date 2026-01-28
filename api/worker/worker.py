import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import onnxruntime as ort

app = FastAPI()

MODEL_SVM_PATH = "modelo_unico.onnx"
CONFIDENCE_THRESHOLD = 0.5

def calculate_vector_angle(a, b, c):
    """Retorna float nativo do Python (não numpy)"""
    pa, pb, pc = np.array(a), np.array(b), np.array(c)
    ba = pa - pb 
    bc = pc - pb 
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return float(round(angle, 2)) 

print("Carregando YOLO...")
model_yolo = YOLO("best.pt")

print("Carregando ONNX...")
ort_session = None
input_name = None
try:
    ort_session = ort.InferenceSession(MODEL_SVM_PATH)
    input_name = ort_session.get_inputs()[0].name
    print("ONNX Carregado.")
except:
    print("AVISO: ONNX não encontrado. Rodando em modo fallback.")

@app.post("/predict")
async def process_image(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model_yolo.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    response = {
        "found": False,
        "exercise": "Ninguem detectado",
        "confidence": 0.0,
        "angles": {},
        "bbox": []
    }

    for result in results:

        if result.boxes is None or result.boxes.shape[0] == 0:
            continue
            
        if result.keypoints is None or result.keypoints.xy.shape[1] == 0:
            continue

        try:

            kps_px = result.keypoints.xy.cpu().numpy()[0]
            kps_xyn = result.keypoints.xyn.cpu().numpy()[0]
            box = result.boxes.xywh.cpu().numpy()[0]

            if len(kps_px) > 11:
                shoulder_ang = calculate_vector_angle(kps_px[11], kps_px[5], kps_px[7])
                elbow_ang = calculate_vector_angle(kps_px[5], kps_px[7], kps_px[9])

                detected_class = "Desconhecido"
                confidence = 0.0

                if shoulder_ang < 30 and elbow_ang > 140:
                    detected_class = "Repouso"
                elif ort_session:
                    try:
                        input_data = kps_xyn.flatten().reshape(1, -1).astype(np.float32)
                        outputs = ort_session.run(None, {input_name: input_data})
                        pred = int(np.argmax(outputs[1][0]))# Força int
                        probs = outputs[1][0]
                        confidence = float(probs[pred]) 
                        
                        classes = {0: "Bicep Curl", 1: "Front Raise", 2: "Press"}
                        detected_class = classes.get(pred, "Outro")
                    except Exception as e:
                        print(f"Erro ONNX: {e}")
                        pass
                
                response = {
                    "found": True,
                    "exercise": str(detected_class),
                    "confidence": float(confidence),
                    "angles": {
                        "shoulder": float(shoulder_ang),
                        "elbow": float(elbow_ang)
                    },
                    "bbox": [int(b) for b in box] 
                }
                break 

        except IndexError:
            continue

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)