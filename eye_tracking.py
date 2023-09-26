import cv2
from ultralytics import YOLO
from pprint import pprint

model = YOLO('eye_tracking.pt')
cap = cv2.VideoCapture(0)


def eye_tracker(img):
    res = model(source=img, show=False, conf=0.7, save=False)
    res = res[0]
    eye_boxes = {}
    for i in range(len(res.boxes)):
        box = res.boxes[i]
        tensor = box.xyxy
        values_list = tensor[0].tolist()
        if len(values_list) != 0:
            eye_boxes = {'x1': int(values_list[0]),
                         'y1': int(values_list[1]),
                         'x2': int(values_list[2]),
                         'y2': int(values_list[3]),
                         'prob': round(box.conf[0].item(), 2)}
    return eye_boxes


while True:
    success, img = cap.read()
    if success:
        results = eye_tracker(img)
        pprint(results)
        if len(results) != 0:
            pupil_coordinates = (int((results['x1'] + results['x2'])/2), int((results['y1'] + results['y2'])/2))
            cv2.circle(img, pupil_coordinates, 3, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Eye Tracking", img)
    cv2.waitKey(1)
