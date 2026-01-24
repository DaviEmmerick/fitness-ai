import numpy as np
import cv2
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import onnxruntime as ort

app = FastAPI()

MODEL_SVM_PATH = "modelo_unico.onnx"
CONFIDENCE_THRESHOLD = 0.4

def calculate_vector_angle(a, b, c):

    pa = np.array(a)
    pb = np.array(b)
    pc = np.array(c)

    ba = pa - pb 
    bc = pc - pb 

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    
    return round(angle, 2)

model_yolo = YOLO("best.pt")

ort_session = None
input_name = None
try:
    ort_session = ort.InferenceSession(MODEL_SVM_PATH)
    input_name = ort_session.get_inputs()[0].name
    print("ONNX Carregado.")
except Exception as e:
    print(f"AVISO: ONNX não encontrado. O worker rodará apenas com regras geométricas.")

@app.post("/predict")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model_yolo.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    response = {
        "found": False,
        "exercise": "Ninguem",
        "angles": {},
        "bbox": []
    }

    for result in results:
        if result.keypoints is not None and result.keypoints.shape[1] > 0:

            kps_px = result.keypoints.xy.cpu().numpy()[0]
            kps_xyn = result.keypoints.xyn.cpu().numpy()[0]
            
            if len(kps_px) > 11:

                shoulder_ang = calculate_vector_angle(kps_px[11], kps_px[5], kps_px[7])
                elbow_ang = calculate_vector_angle(kps_px[5], kps_px[7], kps_px[9])

                detected_class = "Desconhecido"
                
                if shoulder_ang < 30 and elbow_ang > 140:
                    detected_class = "Repouso"
                elif ort_session:
    
                    try:
                        input_data = kps_xyn.flatten().reshape(1, -1).astype(np.float32)
                        outputs = ort_session.run(None, {input_name: input_data})
                        pred = np.argmax(outputs[1][0]) 
                        
                        classes = {0: "Bicep Curl", 1: "Front Raise", 2: "Press"}
                        detected_class = classes.get(pred, "Outro")
                    except:
                        pass
                else:
                    if elbow_ang < 100: detected_class = "Bicep Curl (Provavel)"

                box = result.boxes.xywh.cpu().numpy()[0]
                
                response = {
                    "found": True,
                    "exercise": detected_class,
                    "angles": {
                        "shoulder": shoulder_ang,
                        "elbow": elbow_ang
                    },
                    "bbox": [int(b) for b in box] 
                }
                break 

    return response

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)