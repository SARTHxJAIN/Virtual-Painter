import cv2
import numpy as np
import time
import os
import handTrackingModule as htm

brush = 5
eraser = 20

folderPath = "opencv\\virtualPainter\\Header"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255,0,255)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0

imgCanvas = np.zeros((480,640,3),np.uint8)

while True:
    # import image
    success, img = cap.read()
    img = cv2.flip(img,1)

    # find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        #print(lmList)

        #tip of index & middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        # check which finger are up
        fingers = detector.fingersUp()
        #print(fingers)

        # if selection mode - two finger up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            #print("selection Mode")
            #checking for click
            if y1 < 64:
                if 125<x1<225:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 275<x1<375:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 400<x1<475:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 525<x1<600:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawColor,cv2.FILLED)


        #if draw mode - index finger up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,drawColor,cv2.FILLED)
            #print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraser)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraser)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brush)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brush)

            xp,yp=x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # setting the header image
    img[0:64,0:640] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("image",img)
    cv2.imshow("imageCanvas",imgCanvas)
    cv2.imshow("Inv",imgInv)
    cv2.waitKey(1)