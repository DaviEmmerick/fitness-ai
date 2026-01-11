from ultralytics import YOLO
import cv2
import joblib
import numpy as np
import os

model_yolo_path = "best.pt" # pose-estimation
model_knn_path = "model.pkl" # knn - check pose

if os.path.exists(model_yolo_path):
    print(f"YOLO encontrado: {model_yolo_path}")
    model_yolo = YOLO(model_yolo_path)
else:
    print("'best.pt' não achado. Usando 'yolov8n-pose.pt' padrão.")
    model_yolo = YOLO('yolov8n-pose.pt')

if os.path.exists(model_knn_path):
    print(f"KNN encontrado: {model_knn_path}")
    knn_model = joblib.load(model_knn_path)
else:
    print(f"Erro: {model_knn_path} não encontrado. Treine o modelo primeiro!")
    exit()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model_yolo.predict(frame, classes=[0], conf=0.5, verbose=False)
    
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        
        keypoints = results[0].keypoints.xyn.cpu().numpy()[0] 

        input_data = keypoints.flatten().reshape(1, -1)
        
        prediction = knn_model.predict(input_data)[0]
        probs = knn_model.predict_proba(input_data)[0]
        confidence = max(probs)

        status = "CERTO" if prediction == 1 else "ERRADO"
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255) # Verde ou Vermelho
        
        label = f"{status} ({confidence*100:.0f}%)"
        
        cv2.rectangle(annotated_frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, label, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Academia IA", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()