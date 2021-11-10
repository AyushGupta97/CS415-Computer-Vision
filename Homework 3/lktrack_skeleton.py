#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow starter code. 

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
from time import clock

lk_params = dict(winSize=(2, 2), #You have finetune these
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


feature_params = dict(maxCorners=50,
                      qualityLevel=0.6,
                      minDistance=7,
                      blockSize=15)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        print(video_src)
        self.frame_idx = 0
    
    def draw_str(dst, x, y, s):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    def anorm2(a):
        return (a*a).sum(-1)

    def run(self):
        framec = 0
        ret, frame = self.cam.read()                        
        while True:            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            print(len(self.tracks))

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1]
                                 for tr in self.tracks]).reshape(-1, 1, 2) #points to track
                
                # Fill in code for your Optical Flow algorithms here at p0 points (Part B)
                
                
                #Fill in code to compute new points to track using the flow (Part C)
                new_tracks = [] 
                
                
                
                self.tracks = new_tracks #new points to track from the next frame onwards
                cv2.polylines(vis, [np.int32(tr)
                                    for tr in self.tracks], False, (0, 255, 0), 3)
                draw_str(vis, 20, 20, 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0: #Every few frames we have to recompute features, this includes starting frame
                mask = np.zeros_like(frame_gray) #mask to display features
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                #Fill in code to compute features in the frame, and append to tracks (Part A)

            # Increment frame index and plot
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            fn = "out/lkfilem_absurd"+str(framec).rjust(4, '0')+".png"
            cv2.imwrite(fn, vis)
            framec = framec+1

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            else:
                ret, frame = self.cam.read()
                if frame is None:                    
                    break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
