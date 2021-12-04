#!/usr/bin/env python

# Lucas-Kanade sparse optical flow
# Gives the flow vectors of some "interesting" features
# run with python3 lktrack_sparse.py [video]

import numpy as np
import cv2

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=50,
                      qualityLevel=0.6,
                      minDistance=7,
                      blockSize=15)


# took outside of class b/c errors
def anorm2(a):
    return (a * a).sum(-1)


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        print(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)  # points to track

                # Fill in code for your Optical Flow algorithms here at p0 points (Part B)
                # cv2.calcOpticalFlowPyrLK -> automatically constructs the image pyramid
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1

                # Fill in code to compute new points to track using the flow (Part C)
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len: # make sure num features tracked is less than prespecified length
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks  # new points to track from the next frame onwards
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:  # Every few frames we have to recompute features, this
                # includes starting frame
                mask = np.zeros_like(frame_gray)  # mask to display features
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                # Fill in code to compute features in the frame, and append to tracks (Part A)
                # (compute good features to track every once in a while including the starting frame)
                # For sparse flow, Do not worry about the track length here
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # finds N strongest corners by
                # Shi-Tomasi method
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            # Increment frame index and plot
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
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
