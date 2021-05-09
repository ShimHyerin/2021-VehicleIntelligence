import cv2
import numpy as np
import mediapipe as mp
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

file_list = ['./ok.mp4', './pinky.mp4', './free.mp4', './door.mp4']
with mp_hands.Hands(
    # static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    for idx, file in enumerate(file_list):
      print(file)
      cap = cv2.VideoCapture(file)
      print('cap', cap.isOpened())
      # csv header setting
      csvtmp = []
      filetmp = file.split('/')
      filename = filetmp[1].split('.')[0]
      coordinates = [filename]

      csvtmp.insert(0, 'filename')
      for i in range(1, 22): # hand landmarks = 21개
        csvtmp += ['hand_x{}'.format(i), 'hand_y{}'.format(i), 'hand_z{}'.format(i)]
      with open(filename+'.csv', mode='w', newline='') as f:
        csv_output = csv.writer(f, delimiter =',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        csv_output.writerow(csvtmp)

      frames = []
      fps = 30
      while cap.isOpened():
        success, image = cap.read()
        if not success: break
        image = cv2.resize(image, (540, 960))

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frames.append(image)
        coordinates = [filename]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    # res = (x, y, z)
                    coordinates.append(x)
                    coordinates.append(y)
                    coordinates.append(z)

                mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if len(coordinates) == 127:
            coor1 = coordinates[:64]
            coor2 = coordinates[64:]
            coor2.insert(0, filename)
            with open(filename+'.csv', mode='a', newline='') as f:
                csv_output = csv.writer(f, delimiter =',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
                csv_output.writerow(coor1)
                csv_output.writerow(coor2)
        else: 
            with open(filename+'.csv', mode='a', newline='') as f:
                csv_output = csv.writer(f, delimiter =',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
                csv_output.writerow(coordinates)
        cv2.imshow('data_handDetection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break
      
      size = (540, 960)
      output = cv2.VideoWriter(filename+'_detection.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
      for i in range(len(frames)):
        output.write(frames[i])
      output.release()
      cap.release()


def combineCSV(file_list, name):
    with open(name, 'w') as f:
        for i in range(len(file_list)):
            if i == 0:
                with open(file_list[i], 'r') as f2:
                    while True:
                        line = f2.readline()
                        if not line: break
                        f.write(line)
            else:
                with open(file_list[i], 'r') as f2:
                    n = 0
                    while True:
                        line = f2.readline()
                        if n != 0: f.write(line)
                        if not line: break
                        n += 1

ok = './ok.csv'
free = './free.csv'
pinky = './pinky.csv'
door = './door.csv'
csv_list = [free, ok, pinky, door]
combineCSV(csv_list, 'DATA_ALL.csv')

# combineCSV(ok, free, 'DATA_OK.csv')
# combineCSV(pinky, free, 'DATA_PINKY.csv')
# combineCSV(door, free, 'DATA_DOOR.csv')

