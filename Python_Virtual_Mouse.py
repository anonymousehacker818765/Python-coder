import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpHand = mp.solutions.hands
pro = mpHand.Hands()
tipIds = [4, 8, 12, 16, 20]
plocX, plocY = 0, 0
clocX, clocY = 0, 0
frameR = 110
wScr, hScr = 1280, 720
smoothening = 1
wCam, hCam = 660, 500

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pro.process(imgRGB)
    if result.multi_hand_landmarks:
        for hand1 in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand1, mpHand.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec((0, 255, 255), 3, 2))


            hand_num = 0
            xList = []
            yList = []
            lmList = []
            myHand = result.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            cv2.rectangle(img, (xmin - 23, ymin - 23), (xmax + 23, ymax + 23),
                          (0, 255, 0), 2)

            # print(lmList)

            fingers = []
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

            for id in range(1, 5):

                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                pyautogui.moveTo(clocX * 2, clocY * 2)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                time.sleep(1)
                pyautogui.click()

            if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                time.sleep(1)
                pyautogui.click(clicks=2)

            # print(fingers)
            # print(wScr, hScr)

            if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                time.sleep(1)
                pyautogui.click(button='right')

    cv2.imshow("RRR", img)
    cv2.waitKey(1)
