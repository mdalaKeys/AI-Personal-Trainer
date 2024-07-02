"""
Pose Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""
import math
import cv2
import mediapipe as mp

class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """
    def __init__(self, staticMode=False, modelComplexity=1, smoothLandmarks=True,
                 enableSegmentation=False, smoothSegmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smoothLandmarks,
                                     enable_segmentation=self.enableSegmentation,
                                     smooth_segmentation=self.smoothSegmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][0] - self.lmList[11][0]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][0] - ad
                x2 = self.lmList[15][0] + ad
            else:
                x1 = self.lmList[12][0] - ad
                x2 = self.lmList[11][0] + ad

            y2 = self.lmList[29][1] + ad
            y1 = self.lmList[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1, z1 = p1[:3]
        x2, y2, z2 = p2[:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, img, info

    def findAngle(self, p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        x1, y1, z1 = p1[:3]
        x2, y2, z2 = p2[:3]
        x3, y3, z3 = p3[:3]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, scale // 5))
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1, scale // 5))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), scale + 5, color, max(1, scale // 5))
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale + 5, color, max(1, scale // 5))
            cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), scale + 5, color, max(1, scale // 5))
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, color, max(1, scale // 5))
        return angle, img

    def angleCheck(self, myAngle, targetAngle, offset=20):
        return targetAngle - offset < myAngle < targetAngle + offset

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector(staticMode=False, modelComplexity=1, smoothLandmarks=True,
                            enableSegmentation=False, smoothSegmentation=True,
                            detectionCon=0.5, trackCon=0.5)
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
        if lmList:
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            length, img, info = detector.findDistance(lmList[11], lmList[15], img=img, color=(255, 0, 0), scale=10)
            angle, img = detector.findAngle(lmList[11], lmList[13], lmList[15], img=img, color=(0, 0, 255), scale=10)

            isCloseAngle50 = detector.angleCheck(myAngle=angle, targetAngle=50, offset=10)
            print(isCloseAngle50)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
