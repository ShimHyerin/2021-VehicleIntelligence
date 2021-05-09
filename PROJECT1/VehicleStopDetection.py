# Scikit-learn으로 machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


ok = './DATA_OK.csv'
pinky = './DATA_PINKY.csv'
door = './DATA_DOOR.csv'
d = './DATA_ALL.csv'

def training(csv, state):
    df = pd.read_csv(csv)
    data = df.drop('filename', axis = 1)
    target = df['filename']
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5, random_state = state)

    # 데이터가 특정 카테고리에 속할지 예측하기 위해 로지스틱 회귀 사용
    lr = LogisticRegression()
    model = lr.fit(train_data, train_target)
    return model

# model_ok = training(ok, 1203)
# model_pinky = training(pinky, 934)
# model_door = training(door, 1355)
model = training(d, 1888)

stop = False
chk_door = False
emergencyLights = False

cnt = 0
cnt2 = 0
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # Webcam이므로 continue
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    coordinates = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for data in hand_landmarks.landmark:
                coordinates.append(data.x)
                coordinates.append(data.y)
                coordinates.append(data.z)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if coordinates != [] and len(coordinates) == 63:
        res = pd.DataFrame([coordinates])
        pred = str(model.predict(res)[0])
        if stop == True: # Stop, gesture = ok
            # pred_ok = str(model_ok.predict(res)[0])
            # if pred_ok == 'ok' and chk_door == False: 
            if pred == 'ok' and chk_door == False: 
                cnt += 1
                # cv2.putText(image, pred_ok, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=3)
                cv2.putText(image, pred, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=3)
                cv2.putText(image, str(round(100/90 * cnt, 2))+'%', (190, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),thickness=2)
            else:
                # pred_door = str(model_door.predict(res)[0])
                # if pred_door == 'door': 
                if pred == 'door': 
                    cnt2 += 1
                    # cv2.putText(image, pred_door, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),thickness=3)
                    cv2.putText(image, pred, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),thickness=3)
                    cv2.putText(image, str(round(100/90 * cnt2, 2))+'%', (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),thickness=2)
                if cnt2 == 90:
                    if chk_door==False: chk_door = True
                    else: chk_door = False
                    cnt2 = 0
            if cnt == 90 and chk_door == False:
                stop = False
                emergencyLights = False
                cnt = 0
                cnt2 = 0
        else: # Drive, gesture = Stop
            # pred_pinky = str(model_pinky.predict(res)[0])
            # if pred_pinky == 'pinky': 
            if pred == 'pinky': 
                cnt += 1
                cv2.putText(image, 'stop', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=3)
                cv2.putText(image, str(round(100/90 * cnt, 2))+'%', (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),thickness=2)
            if cnt == 90:
                stop = True
                emergencyLights = True
                cnt = 0

    # Always prints
    if stop == True: 
        currentStop = 'Stop'
        text = 'If you want to drive the car again, take a "OK gesture".'
        cv2.putText(image, text, (130, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 90, 185),thickness=1)
    else: 
        currentStop = 'Go'
        text = 'If you want to stop, take a "stop gesture".'
        cv2.putText(image, text, (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 90, 185),thickness=1)
    if chk_door == True: 
        currentDoor = 'Open'
        text = 'Door must be "Close" before driving...'
        cv2.putText(image, text, (170, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (143, 255, 255),thickness=1)
    else: 
        currentDoor = 'Close'
        text = 'You can open the door by "door gesture"!'
        if stop == True: cv2.putText(image, text, (170, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (143, 255, 255),thickness=1)
    if emergencyLights == True: currentLights = 'On'
    else: currentLights = 'Off'

    currentState = 'driving: {}'.format(currentStop)
    cv2.putText(image, currentState, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),thickness=1)
    currentState = 'door: {}'.format(currentDoor)
    cv2.putText(image, currentState, (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),thickness=1)
    currentState = 'emergency lights: {}'.format(currentLights)
    cv2.putText(image, currentState, (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),thickness=1)

    # show
    cv2.imshow('Vehicle Stop Detection System', image)
    # exit
    if cv2.waitKey(10) & 0xFF == ord('q'): break
cap.release()