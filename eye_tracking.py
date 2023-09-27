import cv2
from ultralytics import YOLO
import dlib

model = YOLO('eye_tracking.pt')
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


def iris_tracker(img, draw=False):
    # YOLO: estrazione x,y delle pupille
    res = model(source=img, show=False, conf=0.7, save=False)
    res = res[0]
    iris_boxes = {}
    pupil_coordinates = []
    for i in range(len(res.boxes)):
        box = res.boxes[i]
        tensor = box.xyxy
        values_list = tensor[0].tolist()
        if len(values_list) != 0:
            iris_boxes = {'x1': int(values_list[0]),
                          'y1': int(values_list[1]),
                          'x2': int(values_list[2]),
                          'y2': int(values_list[3]),
                          'prob': round(box.conf[0].item(), 2)}
        if len(iris_boxes) != 0:
            pupil_coordinates.append(
                (int((iris_boxes['x1'] + iris_boxes['x2']) / 2), int((iris_boxes['y1'] + iris_boxes['y2']) / 2))
            )
            if draw:
                cv2.circle(img, pupil_coordinates[i], 5, (255, 0, 255), cv2.FILLED)
    if len(pupil_coordinates) == 2:
        if pupil_coordinates[0][0] > pupil_coordinates[1][0]:
            pupil_coordinates.reverse()
        print(f'Pupilla destra: x={pupil_coordinates[0][0]}, y={pupil_coordinates[0][1]}')
        print(f'Pupilla sinistra: x={pupil_coordinates[1][0]}, y={pupil_coordinates[1][1]}')
    return pupil_coordinates


def eye_center_tracker(img, draw=False):
    # DLIB: estrazione x,y del centro degli occhi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    center_list = []
    for i, face in enumerate(faces):
        landmarks = predictor(gray, face)
        ext_left = landmarks.part(45)
        int_left = landmarks.part(42)
        ext_right = landmarks.part(36)
        int_right = landmarks.part(39)
        center_left = (int((ext_left.x + int_left.x) / 2), int((ext_left.y + int_left.y) / 2))
        center_right = (int((ext_right.x + int_right.x) / 2), int((ext_right.y + int_right.y) / 2))
        center_list.append(center_right)
        center_list.append(center_left)
        if draw:
            cv2.circle(img, center_left, 5, (0, 0, 255), -1)
            cv2.circle(img, center_right, 5, (0, 0, 255), -1)
    return center_list


def gaze_direction(pupil_coordinates, center_list):
    if len(pupil_coordinates) < 2 or len(center_list) < 2:
        print('###    Non sono riuscito a rilevare entrambi gli occhi.')
        return
    else:
        tol = 2
        if pupil_coordinates[0][0] < (center_list[0][0] - tol) and pupil_coordinates[1][0] < (center_list[1][0] - tol):
            print('###    Stai guardando a DESTRA')
            dir = 'destra'
            cv2.putText(img, text=f'{dir}', org=(50, 50), fontFace=cv2.FONT_ITALIC, fontScale=2, color=(0, 255, 0), thickness=2)
        elif pupil_coordinates[0][0] > (center_list[0][0] + tol) and pupil_coordinates[1][0] > (
                center_list[1][0] + tol):
            print('###    Stai guardando a SINISTRA')
            dir = 'sinistra'
            cv2.putText(img, text=f'{dir}', org=(50, 50), fontFace=cv2.FONT_ITALIC, fontScale=2,
                        color=(255, 0, 0), thickness=2)
        else:
            print('###    Stai guardando al CENTRO')
            dir = 'centro'
            cv2.putText(img, text=f'{dir}', org=(50, 50), fontFace=cv2.FONT_ITALIC, fontScale=2,
                        color=(255, 255, 255), thickness=2)
    return dir


while True:
    success, img = cap.read()
    if success:
        img_h, img_w, img_c = img.shape
        pc = iris_tracker(img, draw=True)
        cl = eye_center_tracker(img)
        print(cl)
        gaze_direction(pc, cl)

    cv2.imshow("Eye Tracking", img)
    cv2.waitKey(1)
