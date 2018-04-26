# elf.tracker = cv2.TrackerMedianFlow_create()
#        self.tracker.init(image=image, boundingBox=box)
# self.tracker = cv2.TrackerBoosting_create()
# small and predictable motion
# Tracking is faster than Detection:
#Therefore, while designing an efficient system usually an object detection is run on every nth frame while the tracking algorithm is employed in the n-1 frames in between
#They, therefore, have more knowledge about the general class of the object. On the other hand, tracking algorithms know more about the specific instance of the class they are tracking.
# Tracking preserves identity: The output of object detection is an array of rectangles that contain the object. However, there is no identity attached to the object. THerefore, traccking is superior
#dont pass in the name of the file, pass in the contents
#model: coordinates, objects, tracker
#view: camera
#controller: unit test (read, write)
'''
step 1) Create object tracker: 
step 2) Open video: 
step 3) Read first frame :
step 4) Define a bounding box for that frame along with a strange step :
step 5) Initialize Tracker with first frame :
step 6) Start tracking:
'''
import cv2
import sys
extra_references = []

class HumanTracker(object):
    video = None
    def __init__(self, videoPath, theObjectTracker=cv2.TrackerMedianFlow_create()):
        global extra_references
        extra_references.append(self)
        self.tracker = theObjectTracker
        self.videoPath = videoPath
     
    def initTracker(self, theVideo, boundingBox=(700, 500, 286, 320)):
        self.video = theVideo
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()
        ok, frame = self.video.read()
        if not ok:
            print("Could not read video file")
            sys.exit()
        self.bbox = boundingBox
        #Allows the user to circle the objects they would like to track
        #self.bbox = cv2.selectROI(frame, False)
        ok = self.tracker.init(frame, self.bbox)

    def updateTracker(self, updateLimit, video):
        i = 0
        print(str(type(video)))
        while i < updateLimit:
            self.ok, self.frame = video.read()
            if not self.ok:  
                break
            self.ok, self.bbox = self.tracker.update(self.frame)
            # Draw bounding box
            if self.ok:
                # Tracking success
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                cv2.rectangle(self.frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(self.frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            cv2.imshow("Tracking", self.frame)
            i  = i + 1

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

