import cv2
import os
import sys
import unittest
import threading

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.tracker

class SingleObjectTrackerTest(unittest.TestCase):
    def setUp(self):
        self.video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'people_walking.mp4'
        )
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        box = (421, 417, 100, 160)
        self.tracker = leapvision.tracker.SingleObjectTracker(
            frame,
            box
        )

    def test_same_image(self):
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 421, delta=2)
        self.assertAlmostEqual(box[1], 417, delta=2)
        self.assertAlmostEqual(box[2], 100, delta=2)
        self.assertAlmostEqual(box[3], 160, delta=2)

    def test_future_frame(self):
        ok, frame = self.camera.read()
        for x in range(30):
            ok, frame = self.camera.read()
            ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 423, delta=2)
        self.assertAlmostEqual(box[1], 325, delta=2)
        self.assertAlmostEqual(box[2], 100, delta=2)
        self.assertAlmostEqual(box[3], 160, delta=2)

    def test_demo(self):
        enabled = bool(os.environ.get('TRACKER_DEMO_ENABLED'))
        if not enabled:
            return
        video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'people_walking.mp4'
        )
        key = 0
        while key & 0xff != ord('q'):
            camera = cv2.VideoCapture(video_path)
            ok, frame = camera.read()
            box = (421, 417, 100, 160)
            self.tracker = leapvision.tracker.SingleObjectTracker(
                frame,
                box
            )
            cv2.namedWindow('camera')
            cv2.moveWindow('camera', 0, 0)
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                tick_count = cv2.getTickCount()
                ok, box = self.tracker.track(frame)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
                if not ok:
                    continue
                cv2.putText(
                    frame,
                    "fps: {}".format(int(fps)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(
                    frame,
                    box[:2],
                    (box[0]+box[2], box[1]+box[3]),
                    (0, 255, 0),
                    2
                )
                cv2.imshow('camera', frame)
                key = cv2.waitKey(1)
            self.tracker.clear()
            
  '''          
class HumanTrackerTest(unittest.TestCase):
    testVideo = None
    #setup
    def setUp(self):
        videoPath = os.path.join(os.path.dirname(__file__), 'videos', '924810169.mp4')
        self.humanTracker = leapvision.tracker.HumanTracker(videoPath)
        print("setUP")
    #Test Creation of Tracker
    def test_trackerConstructor(self):
        self.assertNotEqual(self.humanTracker.tracker, None) 
        self.assertTrue(type(self.humanTracker.videoPath), type(""))
        self.assertTrue(type(self.humanTracker), type(leapvision.tracker.HumanTracker))
        print("trackerConstructor")
    #Test Initialization of Tracker
    def test_initTracker(self):
        self.expectedBbox = (700, 500, 286, 320)
        self.humanTracker.initTracker(cv2.VideoCapture(self.humanTracker.videoPath))
        actualBbox = self.humanTracker.bbox
        self.assertNotEqual(self.humanTracker.video, None)
        self.assertNotEqual(actualBbox, None) 
        self.assertEqual(actualBbox, self.expectedBbox)
        self.testVideo = self.humanTracker.video
        self.test_updateTracker(self.testVideo, 1)
        print(threading.active_count())
    def test_updateTracker(self, theVideo = None, permission = 0):
        print(threading.active_count())
        framesToCycleThrough = 48
        if permission == 1:
            self.humanTracker.updateTracker(framesToCycleThrough, theVideo)
            '''


if __name__ == '__main__':
    unittest.main()
