import cv2
from ultralytics import YOLO
from pprint import pprint

model = YOLO('eye_tracking.pt')
cap = cv2.VideoCapture(0)


def eye_tracker(img):
    results = model(source=img, show=False, conf=0.7, save=False)
    return results[0]


while True:
    success, img = cap.read()
    if success:
        results = eye_tracker(img)
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            tensor = box.xyxy
            values_list = tensor[0].tolist()
            if len(values_list) != 0:
                eye_boxes = {'x1': values_list[0], 'y1': values_list[1], 'x2': values_list[2], 'y2': values_list[3]}
                pprint(eye_boxes)

            #print(values_list)
            #print("Probability:", box.conf[0].item())

    cv2.imshow("Eye Tracking", img)
    cv2.waitKey(1)
