import cv2
import joblib
import numpy as np
from ultralytics import YOLO
from collections import deque

MODEL_KNN_PATH = "modelo_unico.pkl"  
MODEL_YOLO_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 7

EXERCISE_RULES = {
    "Bicep curl": {"joint": "elbow", "start": 150, "end": 60, "type": "decreasing"}, 
    "Front raise": {"joint": "shoulder", "start": 25, "end": 80, "type": "increasing"}, 
    "Shoulder press": {"joint": "elbow", "start": 70, "end": 150, "type": "increasing"}, 
}

CLASS_DECODER = {
    0: ("Bicep curl", "CORRIGIR POSTURA", (0, 0, 255)),   
    1: ("Bicep curl", "PERFEITO", (0, 255, 0)),           
    2: ("Front raise", "CORRIGIR POSTURA", (0, 0, 255)),
    3: ("Front raise", "PERFEITO", (0, 255, 0)),
    4: ("Shoulder press", "CORRIGIR POSTURA", (0, 0, 255)),
    5: ("Shoulder press", "PERFEITO", (0, 255, 0)),
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

try:
    print(f"Carregando {MODEL_KNN_PATH}...")
    knn_model = joblib.load(MODEL_KNN_PATH)
    model_yolo = YOLO(MODEL_YOLO_PATH)
    print("Sistema pronto!")

except Exception as e:
    print(f"Erro ao carregar: {e}")
    model_yolo = YOLO("yolov8n-pose.pt")

history = deque(maxlen=SMOOTHING_WINDOW)
current_stage = "START"
reps = 0
last_ex = ""

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break

    results = model_yolo.predict(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False, stream=True)
    
    for result in results:
        annotated_frame = frame.copy()
        
        if result.keypoints is not None and len(result.keypoints) > 0:
            kps_xyn = result.keypoints.xyn.cpu().numpy()[0]
            kps_px = result.keypoints.xy.cpu().numpy()[0]  

            if len(kps_px) > 11:
                input_data = kps_xyn.flatten().reshape(1, -1)
                
                try:
                    pred_raw = knn_model.predict(input_data)[0]
                    probs = knn_model.predict_proba(input_data)[0]
                    confidence = max(probs)
                    
                    history.append(pred_raw)
                    final_pred = max(set(history), key=history.count)
                    
                    ex_name, status_text, status_color = CLASS_DECODER.get(final_pred, ("Analisando...", "", (255,255,255)))

                except ValueError:
                    ex_name = "ERRO MODELO"
                    status_text = "Use modelo 34 features"
                    status_color = (0,0,255)
                    confidence = 0

                if ex_name in EXERCISE_RULES:
                    if ex_name != last_ex:
                        reps = 0
                        current_stage = "START"
                        last_ex = ex_name
                    
                    rule = EXERCISE_RULES[ex_name]
                    angle = 0
                    if rule["joint"] == "elbow":
                        angle = calculate_angle(kps_px[5], kps_px[7], kps_px[9])
                    elif rule["joint"] == "shoulder":
                        angle = calculate_angle(kps_px[11], kps_px[5], kps_px[7])
                    
                    if rule["type"] == "increasing":
                        if angle < rule["start"] + 15: current_stage = "START"
                        if angle > rule["end"] - 10 and current_stage == "START":
                            current_stage = "END"; reps += 1
                    else:
                        if angle > rule["start"] - 15: current_stage = "START"
                        if angle < rule["end"] + 10 and current_stage == "START":
                            current_stage = "END"; reps += 1

                h, w, _ = annotated_frame.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.rectangle(annotated_frame, (10, h-80), (300, h-10), (0,0,0), -1)
                
                cv2.putText(annotated_frame, ex_name, (20, h-50), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.putText(annotated_frame, f"{status_text} ({int(confidence*100)}%)", (20, h-20), 
                            font, 0.6, status_color, 1, cv2.LINE_AA)
                
                cv2.putText(annotated_frame, f"REPS: {reps}", (w - 180, 50), 
                            font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

                for idx in [5, 7, 9, 11]: 
                    if idx < len(kps_px):
                        cv2.circle(annotated_frame, (int(kps_px[idx][0]), int(kps_px[idx][1])), 5, status_color, -1)

        cv2.imshow("Academia IA", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()