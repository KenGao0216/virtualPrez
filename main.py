import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
#cam
width, height = 1280, 720
folderPath = "Presentation" #pres slideshow in this folder
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4, height)

pathImages = sorted(os.listdir(folderPath), key = len)

imgNumber = 0
hs, ws = 120, 213 #facecam
gestureThreshold = 400
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False


#hand
detector = HandDetector(detectionCon=0.8, maxHands = 1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #horizontal flip of cam
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 0, 255), 10)
    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        #limit pointer area
        xVal = int(np.interp(lmList[8][0], [width/2, width],[0, width]))
        yVal = int(np.interp(lmList[8][0], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            annotationStart = False
            #1-go back
            if fingers == [1,0,0,0,0]:
                annotationStart = False
                if imgNumber>=0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber -=1
            #2 - go next
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                if imgNumber < len(pathImages)-1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        # 3- pointer
        if fingers == [0,1,1,0,0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False
        # 4-draw
        if fingers == [0,1,0,0,0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber+=1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False
        # 5 -erase
        if fingers == [0,1,1,1,0]:
            if annotations:
                if annotationNumber> -1:
                    annotations.pop(-1)
                    annotationNumber =0
                    buttonPressed = True
    else:
        annotationStart = False

    if buttonPressed:
        buttonCounter +=1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range (len(annotations)):
        for j in range (len(annotations[i])):
            if i!= 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 8)



    imgSmall = cv2.resize(img, (ws, hs))
    h,w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)


    key =cv2.waitKey(1)
    if key ==ord('q'):
        break


