import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, numhands=2, detectionConfidence=0.5, complexity=1, trackingConfidence=0.5):
        self.mode = mode
        self.numhands = numhands
        self.detectConfidence = detectionConfidence
        self.modelComplex = complexity
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.Fingertipspos = [4, 8, 12, 16, 20]
        self.hands = self.mpHands.Hands(self.mode, self.numhands, self.modelComplex,
                                        self.detectConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detectHands(self, img, draw=False):
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_color)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img


    def HandPosition(self, img, handNo=0):
        self.fingertips = []

        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            # Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h, w, c = img.shape
                # find the position
                cx, cy = int(lm.x * w), int(lm.y * h)  # center
                self.fingertips.append([id, cx, cy])

        return self.fingertips

    def NumFingers(self):
        fingers_counter = []
        if self.fingertips[self.Fingertipspos[0]][1] < self.fingertips[self.Fingertipspos[0] - 1][1]:
            fingers_counter.append(1)
        else:
            fingers_counter.append(0)
        for id in range(1, 5):
            if self.fingertips[self.Fingertipspos[id]][2] < self.fingertips[self.Fingertipspos[id] - 2][2]:
                fingers_counter.append(1)
            else:
                fingers_counter.append(0)
        return fingers_counter



