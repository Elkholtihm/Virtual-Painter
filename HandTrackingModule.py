import mediapipe as mp
import time
import cv2
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandDetector():
    def __init__(self, mode =False, maxHand=2,  detectionCon = 0.5, trackCon = 0.5, complexity=1):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity
        # create an object of hands (its like formality that we need to use before starting)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,  
            max_num_hands=self.maxHand, 
            min_detection_confidence=self.detectionCon,  
            min_tracking_confidence=self.trackCon,  
            model_complexity=self.complexity,
            )

        # draw landmarks using mediapipe
        self.mpDraw = mp.solutions.drawing_utils
    def Track(self, img, draw_circles = False):
        # convert image color to supported  format by mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # processing image by mediapipe, here we're using the object weve created previously 
        self.results = self.hands.process(imgRGB)

        #create a list to store data about landmarks position
        lmlist = []

        if self.results.multi_hand_landmarks:
            # iterate over all hands landmarks appeared
            for handlms in self.results.multi_hand_landmarks:
                # draw landmarks on image using 'handlms' and connect between these landmarks
                self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

                # iterate over a hand at a time to draw a circle around landmarks
                for id, lm in enumerate(handlms.landmark):

                    #get the height, width and the channels of the image
                    h, w, c = img.shape

                    # take the x and y position of each landmark 
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmlist.append([id, cx, cy])

                    if draw_circles:
                    # draw a circle in each landmark
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return img, lmlist
    def RaisedFingers(self, lmlist):
        IdList = [8, 12, 16, 20]
        fingers = np.zeros(shape = 5, dtype=int)
        for i in IdList:
            if lmlist[i][2] < lmlist[i-1][2]:
                fingers[IdList.index(i) + 1] = 1
        if lmlist[4][1] < lmlist[3][1]:
            fingers[0] = 1
        
        return fingers
        




