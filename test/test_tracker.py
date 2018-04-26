import cv2
import os
import sys
import unittest
import threading

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.tracker

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
    
if __name__ == '__main__':
    unittest.main()
    


        

