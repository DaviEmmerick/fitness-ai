from ultralytics import YOLO
import cv2
import os


model_path = "best.pt"

if os.path.exists(model_path):
    print(f"O modelo {model_path} existe e foi encontrado")
    model = YOLO(model_path)
else:
    print(f"O modelo {model_path} não foi encontrado")
    exit()

cap = cv2.VideoCapture(0)

while True:
    success, frames = cap.read()

    if not success:
        break

    results = model.predict(frames, classes = [0], conf = 0.5, verbose = True)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO pose estimation - gym", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()