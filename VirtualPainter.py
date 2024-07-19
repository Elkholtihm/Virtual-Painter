import HandTrackingModule as hm
import time
import cv2
import numpy as np
import os


# capturing videos via the webcam (used to open a video file or webcam)
cap = cv2.VideoCapture(0)
w, h = 1280, 720
cap.set(3, w) # width
cap.set(4, h) # height

# calculating number of frame per second
pTime = 0

# get images and store its pixels in list
images_names = os.listdir(r'selection')
images = []
for name in images_names:
    image = cv2.imread(os.path.join(r'selection', name))
    resized = cv2.resize(image, (w, 125))
    images.append(resized)

# initialize image in the header
header = images[0]

# define instance of the class
detector = hm.HandDetector(maxHand = 1, detectionCon=0.8)

# intilize the color 
color = (0, 0, 255)

# create another img
imgCanvas = np.zeros((h, w, 3), np.uint8)

mode = 'not detected'
fingers = ['not detected']
while True:

    # read  frames and flip the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # get landmarks
    img, lmlist = detector.Track(img)
    
    #show first image in the top of the frame
    img[0:125, 0:w] = header  

    xp, yp = 0, 0

    # if index finger is up ---> writing
    if len(lmlist) != 0:

        # get cordinate of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)  
        
        # intilize varible to store mode
        mode = ''

        # get raised index 
        fingers = detector.RaisedFingers(lmlist)

        # if index and middle is up ---> selection
        if fingers[1] and fingers[2]:
            # draw a rectangle to indicate selecting mode
            cv2.rectangle(img, (x1, y1-10), (x2, y2), color, 2, cv2.FILLED)
            mode += 'selection'

            # change color and header image based to index finger cordinates
            if 30 < y1 and y1 < 105:
                if x1 > 40 and x1 < 140:
                    color = (0, 255, 0)
                    header = images[0]
                if x1 > 350 and x1 < 440:
                    color = (0, 0, 255)
                    header = images[1]
                if x1 > 700 and x1 < 772:
                    color = (255, 0, 0)
                    header = images[2]
                if x1 > 1050 and x1 < 1160:
                    color = (0, 0, 0)
                    header = images[3]




        # if only index up ---> 
        if fingers[1] and fingers[2] == False:
            mode += 'writing'
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)

            # drawing 
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if color == (0, 0, 0):
                thickness = 50
            else:
                thickness = 30

            cv2.line(img, (xp, yp), (x1, y1), color, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), color, thickness)
            xp, yp = x1, y1
        
    # convert imgCanva from BGR to Grayscale
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # pixels above 50 is set to 0 and 155 if else
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    # convert it to an image with three channels for colors
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # pixel wise AND operation (pixel is white if in both is white else black )
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.putText(img, 'mode : ' + mode, (10, 680), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
    cv2.putText(img, f'{fingers}', (10, 700), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
    # display header image in the top of the frame 
    img[0:125, 0:w] = header

    # calculating the frame rate 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # display the frame rate
    cv2.putText(img, f'fps : {int(fps)}', (1200, 700), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
    # display
    cv2.imshow('Image', img)
    cv2.imshow('canvas', imgCanvas)
    cv2.imshow('Inv', imgInv)
    cv2.waitKey(1)