import cv2
import numpy as np
import time
import PoseModule as pm
from HandTrackingModule import HandDetector

# Initialize video capture, pose detector, and hand detector
cap = cv2.VideoCapture(0)
pose_detector = pm.PoseDetector()
hand_detector = HandDetector(detectionCon=0.8, trackCon=0.8)

# Exercise types
exercises = ["Right Arm Curl", "Left Arm Curl", "Squat", "Push-up"]
exercise_index = -1  # No exercise active at start

# Counters and directions for exercises
counts = [0, 0, 0, 0]
dirs = [0, 0, 0, 0]
pTime = 0

# Set the smaller resolution for the webcam feed
width, height = 640, 360

# Load and resize the button image
button_img = cv2.imread('button.png')
button_img = cv2.resize(button_img, (50, 50))
button_h, button_w, _ = button_img.shape

# Function to toggle exercises
def toggle_exercise():
    global exercise_index
    exercise_index = (exercise_index + 1) % (len(exercises) + 1)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (width, height))
    img = pose_detector.findPose(img, False)
    lmList, bboxInfo = pose_detector.findPosition(img, False)

    # Find hands and their landmarks
    img = hand_detector.findHands(img, draw=True)
    hand_lmList, hand_bboxs = hand_detector.findPosition(img, draw=False)

    # Draw the button on the top right corner
    img[0:button_h, width-button_w:width] = button_img

    if hand_lmList:
        # Get the tip of the index finger (assuming lmList[8] is the index finger tip)
        x, y = hand_lmList[8][0], hand_lmList[8][1]

        # Check if the index finger tip is within the button area
        if width-button_w < x < width and 0 < y < button_h:
            toggle_exercise()
            time.sleep(0.3)  # To avoid multiple toggles in quick succession

    if exercise_index == len(exercises):
        # Display the report at the end
        cv2.putText(img, "Exercise Report", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        for i, exercise in enumerate(exercises):
            cv2.putText(img, f'{exercise}: {int(counts[i])}', (50, 100 + i*40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    elif exercise_index >= 0:
        exercise = exercises[exercise_index]
        cv2.rectangle(img, (70, 50), (495, 75), (255,255,255), cv2.FILLED)
        cv2.putText(img, f'Current Exercise: {exercise}', (70, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)


        if lmList:
            if exercise == "Right Arm Curl":
                # Right Arm Curl
                angle, img = pose_detector.findAngle(lmList[12], lmList[14], lmList[16], img=img)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (height-50, 50))
                color = (0, 255, 0)  # Green color for a gym environment

                if per == 100:
                    color = (0, 0, 255)  # Red color for full extension
                    if dirs[0] == 0:
                        counts[0] += 0.5
                        dirs[0] = 1
                if per == 0:
                    color = (0, 0, 255)  # Red color for full contraction
                    if dirs[0] == 1:
                        counts[0] += 0.5
                        dirs[0] = 0

                # Draw Bar
                cv2.rectangle(img, (50, 50), (10, height - 70), color, 3)
                cv2.rectangle(img, (50, int(bar)), (10, height - 70), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.putText(img, f'R: {int(counts[0])}', (500, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            elif exercise == "Left Arm Curl":
                # Left Arm Curl
                angle, img = pose_detector.findAngle(lmList[15], lmList[13], lmList[11], img=img)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (height - 50, 50))
                color = (0, 255, 0)  # Green color for a gym environment

                if per == 100:
                    color = (0, 0, 255)  # Red color for full extension
                    if dirs[1] == 0:
                        counts[1] += 0.5
                        dirs[1] = 1
                if per == 0:
                    color = (0, 0, 255)  # Red color for full contraction
                    if dirs[1] == 1:
                        counts[1] += 0.5
                        dirs[1] = 0

                # Draw Bar
                cv2.rectangle(img, (50, 50), (10, height - 70), color, 3)
                cv2.rectangle(img, (50, int(bar)), (10, height - 70), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.putText(img, f'L: {int(counts[1])}', (500, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            elif exercise == "Squat":
                # Squats
                # Squats
                angle, img = pose_detector.findAngle(lmList[24], lmList[26], lmList[28], img=img)
                per = np.interp(angle, (250, 160), (0, 100))  # Reverse the interpolation range
                bar = np.interp(angle, (250, 160), (height - 50, 50))  # Reverse the bar position
                color = (0, 255, 0)  # Green color for a gym environment

                if per >= 90:  # Adjust threshold for full extension
                    color = (0, 0, 255)  # Red color for full extension
                    if dirs[2] == 0:
                        counts[2] += 0.5
                        dirs[2] = 1
                elif per <= 10:  # Adjust threshold for full contraction
                    color = (0, 0, 255)  # Red color for full contraction
                    if dirs[2] == 1:
                        counts[2] += 0.5
                        dirs[2] = 0

                # Draw Bar
                cv2.rectangle(img, (50, 50), (10, height - 70), color, 3)
                cv2.rectangle(img, (50, int(bar)), (10, height - 70), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.putText(img, f'S: {int(counts[2])}', (500, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            elif exercise == "Push-up":
                # Push-ups
                angle, img = pose_detector.findAngle(lmList[15], lmList[13], lmList[11], img=img)
                per = np.interp(angle, (217, 280), (0, 100))
                bar = np.interp(angle, (217, 280), (height - 70, 50))  # Adjusted bar range and coordinates
                color = (0, 255, 0)  # Green color for a gym environment

                if per >= 90:  # Adjust threshold for full extension
                    color = (0, 0, 255)  # Red color for full extension
                    if dirs[3] == 0:
                        counts[3] += 0.5
                        dirs[3] = 1
                elif per <= 10:  # Adjust threshold for full contraction
                    color = (0, 0, 255)  # Red color for full contraction
                    if dirs[3] == 1:
                        counts[3] += 0.5
                        dirs[3] = 0

                # Draw Bar
                cv2.rectangle(img, (50, 50), (10, height - 70), color, 3)
                cv2.rectangle(img, (50, int(bar)), (10, height - 70), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.putText(img, f'S: {int(counts[3])}', (500, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Display the frame
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
