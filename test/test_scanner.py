import cv2
import os
import sys
import unittest
import numpy as np

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.scanner


class HumanScannerTest(unittest.TestCase):
    def setUp(self):
        self.scanner = leapvision.scanner.HumanScanner()

    def test_initial_state(self):
        self.assertEqual(self.scanner.scoreThreshold, 0.6)
        self.assertEqual(self.scanner.winStride, (8, 8))
        self.assertEqual(self.scanner.padding, (8, 8))
        self.assertEqual(self.scanner.scale, 1.05)
        self.assertIsNotNone(self.scanner.hog)

    def test_single_person(self):
        test_image_path = os.path.join(
            os.path.dirname(__file__), 'data', 'person_view.jpg')
        test_image = cv2.imread(test_image_path)
        result = self.scanner.scan(test_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'person')
        self.assertAlmostEqual(result[0]['score'], 0.89, delta=1)
        self.assertEqual(result[0]['box'], (2138, 437, 1319, 2638))

    def test_demo(self):
        enabled = bool(os.environ.get('SCANNER_PERSON_DEMO_ENABLED'))
        if not enabled:
            return
        self.scanner = leapvision.scanner.HumanScanner(scoreThreshold=0.75)
        video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'people_walking.mp4'
        )
        key = 0
        while key & 0xff != ord('q'):
            camera = cv2.VideoCapture(video_path)
            cv2.namedWindow('camera')
            cv2.moveWindow('camera', 0, 0)
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                tick_count = cv2.getTickCount()
                result = self.scanner.scan(frame)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
                cv2.putText(
                    frame,
                    "fps: {}".format(int(fps)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )
                for entry in result:
                    box = entry['box']
                    pt1 = (box[0], box[1])
                    pt2 = (pt1[0]+box[2], pt1[1]+box[3])
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                    text = "Score : {}".format(entry['score'])
                    cv2.putText(frame, text, pt1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
                cv2.putText(
                    frame,
                    "fps: {}".format(int(fps)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )
                cv2.imshow('camera', frame)
                key = cv2.waitKey(1)


class FaceScannerTest(unittest.TestCase):
    def setUp(self):
        self.scanner = leapvision.scanner.FaceScanner()

    def test_first_frame(self):
        test_video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'faces.mp4'
        )
        camera = cv2.VideoCapture(test_video_path)
        (ok, frame) = camera.read()
        self.assertTrue(ok)
        faces = self.scanner.scan(frame)
        self.assertEqual(len(faces), 7)
        self.assertEqual(faces[0]['box'], (157, 357, 44, 44))
        

    def test_demo(self):
        enabled = bool(os.environ.get('SCANNER_FACE_DEMO_ENABLED'))
        if not enabled:
            return
        cv2.namedWindow('camera')
        cv2.moveWindow('camera', 0, 0)
        test_video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'faces.mp4'
        )
        camera = cv2.VideoCapture(test_video_path)
        while camera.isOpened():
            (ok, frame) = camera.read()
            if not ok:
                break
            tick_count = cv2.getTickCount()
            faces = self.scanner.scan(frame)
            foreground_mask = np.zeros(frame.shape[:2], np.uint8)
            for entry in faces:
                box = entry['box']
                pt1 = (box[0], box[1])
                pt2 = (pt1[0]+box[2], pt1[1]+box[3])
                cv2.rectangle(foreground_mask, pt1, pt2, (255), cv2.FILLED)

            # TODO: Optimize all this - it's likely making things slow.
            pixelated = cv2.resize(frame, (0,0), fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
            pixelated = cv2.resize(pixelated, (0,0), fx=20.0, fy=20.0, interpolation=cv2.INTER_NEAREST)
            foreground = cv2.bitwise_and(pixelated, pixelated, mask=foreground_mask)
            background_mask = cv2.bitwise_not(foreground_mask)
            background = cv2.bitwise_and(frame, frame, mask=background_mask)
            masked_frame = cv2.add(background, foreground)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
            cv2.putText(
                masked_frame,
                "fps: {}".format(int(fps)),
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
            cv2.imshow('camera', masked_frame)
            key = cv2.waitKey(1)


if __name__ == '__main__':
    unittest.main()
