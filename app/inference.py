import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import onnxruntime as ort
import os
import time

MODEL_SVM_PATH = "modelo_unico.onnx"
MODEL_YOLO_PATH = "best.onnx" 
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 12 

REST_SHOULDER_ANGLE = 25  
REST_ELBOW_ANGLE = 140    

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

def get_geometric_correction(kps_px, predicted_class, shoulder_ang, elbow_ang):

    if predicted_class == "Bicep curl":
        if shoulder_ang > 45: 
            return "Front raise"

    elif predicted_class == "Front raise":
        if elbow_ang < 120: 
            return "Bicep curl"

    return predicted_class

ort_session = None
input_name = None



try:
    if os.path.exists(MODEL_SVM_PATH):
        ort_session = ort.InferenceSession(MODEL_SVM_PATH)
        input_name = ort_session.get_inputs()[0].name
        print("Classificador ONNX carregado!")
    else:
        print(f"Arquivo não encontrado: {MODEL_SVM_PATH}")
except Exception as e:
    print(f"Erro ao carregar ONNX: {e}")

try:
    model_yolo = YOLO(MODEL_YOLO_PATH)
except:
    print("Usando YOLO padrão...")
    model_yolo = YOLO("yolov8n-pose.pt")

history = deque(maxlen=SMOOTHING_WINDOW)

current_exercise = "Aguardando..." 
current_stage = "START"
reps = 0

change_exercise_counter = 0
CHANGE_THRESHOLD = 10  

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

            for idx in [5, 7, 9, 11]: 
                if idx < len(kps_px):
                    cv2.circle(annotated_frame, (int(kps_px[idx][0]), int(kps_px[idx][1])), 5, (0,255,255), -1)

            if len(kps_px) > 11:

                shoulder_ang = calculate_angle(kps_px[11], kps_px[5], kps_px[7])
                elbow_ang = calculate_angle(kps_px[5], kps_px[7], kps_px[9])

                is_resting = (shoulder_ang < REST_SHOULDER_ANGLE) and (elbow_ang > REST_ELBOW_ANGLE)

                detected_class = "Desconhecido"
                confidence = 0
                status_text = ""
                status_color = (255,255,255)

                if not is_resting or current_exercise == "Aguardando...":
                    input_data = kps_xyn.flatten().reshape(1, -1).astype(np.float32)

                    if ort_session is not None:
                        try:
                            outputs = ort_session.run(None, {input_name: input_data})
                            pred_raw = outputs[0][0]
                            probs_dict = outputs[1][0]
                            confidence = probs_dict[pred_raw]
                            
                            temp_class, status_text, status_color = CLASS_DECODER.get(pred_raw, ("Desconhecido", "", (255,255,255)))
                            
                            detected_class = get_geometric_correction(kps_px, temp_class, shoulder_ang, elbow_ang)

                        except Exception as e:
                            detected_class = "ERRO"
                    else:
                        detected_class = "SEM MODELO"
                else:
                    detected_class = current_exercise 


                history.append(detected_class)
                smoothed_class = max(set(history), key=history.count)


                if is_resting:
                    change_exercise_counter = 0 
                    status_text = "REPOUSO / ESPERA"
                    status_color = (200, 200, 200)
                
                else:
                    if smoothed_class != current_exercise and smoothed_class in EXERCISE_RULES:
                        change_exercise_counter += 1
                        
                        if change_exercise_counter > CHANGE_THRESHOLD:
                            current_exercise = smoothed_class
                            reps = 0
                            current_stage = "START"
                            change_exercise_counter = 0
                            print(f"-> Troca de exercício confirmada: {current_exercise}")
                    else:
                        change_exercise_counter = 0 

                if current_exercise in EXERCISE_RULES:
                    rule = EXERCISE_RULES[current_exercise]
                    
                    target_angle = elbow_ang if rule["joint"] == "elbow" else shoulder_ang
                    
                    if rule["type"] == "increasing":
                        if target_angle < rule["start"] + 15: current_stage = "START"
                        if target_angle > rule["end"] - 10 and current_stage == "START":
                            current_stage = "END"; reps += 1
                    else:
                        if target_angle > rule["start"] - 15: current_stage = "START"
                        if target_angle < rule["end"] + 10 and current_stage == "START":
                            current_stage = "END"; reps += 1

                h, w, _ = annotated_frame.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.rectangle(annotated_frame, (10, h-90), (450, h-10), (20,20,20), -1)
                
                cv2.putText(annotated_frame, current_exercise, (20, h-55), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(annotated_frame, f"{status_text}", (20, h-25), font, 0.6, status_color, 1, cv2.LINE_AA)
                
                ind_color = (100, 100, 100) if is_resting else (0, 255, 0)
                ind_text = "HOLD" if is_resting else "ACTIVE"
                cv2.circle(annotated_frame, (w - 40, 40), 10, ind_color, -1)
                cv2.putText(annotated_frame, ind_text, (w - 85, 40), font, 0.5, ind_color, 1, cv2.LINE_AA)
                
                cv2.putText(annotated_frame, f"REPS: {reps}", (w - 220, 80), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    
        cv2.imshow("Academia IA", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()