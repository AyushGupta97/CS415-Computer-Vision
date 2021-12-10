import cv2
import numpy as np
import FingerTracer as ft
from keras.models import load_model
from datetime import datetime

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    canvas_bg = np.zeros([512, 512, 3], dtype=np.uint8)
    canvas_bg.fill(255)
    detector = ft.handDetector(detectionConfidence=0.85)
    prev_x, prev_y = 0, 0
    success, prev_frame = camera.read()
    model = load_model("thumpsUP_model.h5")
    mask = cv2.createBackgroundSubtractorMOG2()
    save_count = 0
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        hand_detect = detector.detectHands(frame)
        fingers = detector.HandPosition(hand_detect)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        thresh = mask.apply(frame)
        thresh = cv2.resize(thresh, [100, 120])
        thresh = thresh.reshape(1, 100, 120, 1)
        gesture_predict = np.argmax(model.predict_on_batch(thresh), axis=1)

        if len(fingers) != 0:
            f1_x, f1_y = fingers[8][1:]
            f2_x, f2_y = fingers[12][1:]
            fin = detector.NumFingers()
            options = dict(color=(255, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=1,
                           line_type=8)
            if sum(fin[1:]) == 1:
                cv2.drawMarker(hand_detect, (f1_x, f1_y), (255, 0, 0), 0, 30, 4)
                if prev_y == 0 and prev_x == 0:
                    prev_x, prev_y = f1_x, f1_y
                cv2.line(canvas_bg, (prev_x, prev_y), (f1_x, f1_y), (0, 0, 0), 15)
            if sum(fin[1:]) == 2:
                cv2.drawMarker(hand_detect, (f1_x, f1_y), (255, 0, 0), 0, 30, 4)
            if sum(fin[1:]) == 3:
                cv2.drawMarker(hand_detect, (f1_x, f1_y), (255, 0, 0), 0, 30, 4)
                if prev_y == 0 and prev_x == 0:
                    prev_x, prev_y = f1_x, f1_y
                cv2.line(canvas_bg, (prev_x, prev_y), (f1_x, f1_y), (255, 255, 255), 15)
            if (gesture_predict[0] != 1) and fin[0]:
                save_count += 1
                if save_count > 30:
                    dt = str(datetime.now())
                    dt = dt.replace(":", "")
                    dt = dt.replace(".", "")
                    filename = "canvas_" + dt + ".jpg"
                    cv2.imwrite(filename, canvas_bg)
                    print("Saving Canvas...")
                    save_count = 0

            prev_x, prev_y = f1_x, f1_y
        cv2.imshow("Camera", hand_detect)
        cv2.imshow('Canvas', canvas_bg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()