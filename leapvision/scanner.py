import cv2
import numpy as np
import os
import foundation


def package_rects_and_weights(rects, weights, scoreThreshold=0.6):
    result = list()
    for (rect, score) in zip(rects, weights):
        if score[0] < scoreThreshold or score[0] > 1.0:
            continue
        result.append({
            'score': score[0],
            'box': tuple(rect)
        })
    result = foundation.non_max_suppression(result)
    return result


class HumanScanner(object):
    '''
    Scanner implementation that returns a list of bounding boxes for any
    humans detected in a given image.
    '''

    def __init__(self, scoreThreshold=0.6, winStride=(8, 8), padding=(8, 8), scale=1.05):
        self.scoreThreshold = scoreThreshold
        self.winStride = winStride
        self.padding = padding
        self.scale = scale
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def scan(self, image):
        (rects, weights) = self.hog.detectMultiScale(
            image,
            winStride=self.winStride,
            padding=self.padding,
            scale=self.scale
        )
        result = list()
        for (rect, score) in zip(rects, weights):
            if score[0] < self.scoreThreshold or score[0] > 1.0:
                continue
            result.append({
                'label': 'person',
                'score': score[0],
                'box': tuple(rect)
            })
        result = foundation.non_max_suppression(result)
        return sorted(result, key=lambda x: x['score'], reverse=True)


class FaceScanner(object):
    def __init__(self):
        weights_path = os.path.join(
            cv2.__path__[0],
            'data',
            'haarcascade_frontalface_default.xml'
        )
        self.frontalface = cv2.CascadeClassifier(weights_path)
        weights_path = os.path.join(
            cv2.__path__[0],
            'data',
            'haarcascade_profileface.xml'
        )
        self.profileface = cv2.CascadeClassifier(weights_path)

    def scan(
        self,
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    ):
        result = list()
        for rect in self.frontalface.detectMultiScale(
            image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=flags,
        ):
            result.append({
                'label': 'face',
                'score': 0.99,
                'box': tuple(rect)
            })
        
        for rect in self.profileface.detectMultiScale(
            image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=flags,
        ):
            result.append({
                'label': 'face',
                'score': 0.99,
                'box': tuple(rect)
            })
        return foundation.non_max_suppression(result)
