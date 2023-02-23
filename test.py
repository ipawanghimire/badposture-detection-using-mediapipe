import cv2
import mediapipe as mp
import numpy as np
import math


# def calc_angle(a, b, c):
#   a = np.array(a)
#  b = np.array(b)
# c = np.array(c)

# midpoint btwn two shoulders:
#ma = (a[0]+b[0])/2
#mb = (a[1]+b[1])/2

# dist_btwn nose and midpoint of two shoulders:
#length = (((m[0]-c[0]) ^ 2)+((m[1]-c[1]) ^ 2)) ^ (1/2)

# calc angle between nose midpoint and left shoulder
#rad = np.arctan2(c[1]-mb, c[0]-ma)-np.arctan2(a[1]-bb, a[0]-aa)
#angle = np.abs(rad*180/np.pi)
# print(angle)

# return (angle)


# calculate length for a good posture
cap = cv2.VideoCapture(0)
#ret, frame = cap.read()

while(True):
    ret, frame = cap.read()
    cv2.imshow('img1', frame)

    # cv2.putText(frame, "press 'Y' to click the picture", (0, 225),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('c1.jpg', frame)
        cv2.destroyAllWindows()
        break
cap.release()


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)as pose:

    m = np.array([0, 0])

    image = cv2.imread('c1.jpg', 0)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    try:
        landmarks = results.pose_landmarks.landmark
        # print(landmarks)

        ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        # midpoint btwn two shoulders:
        ma = (ls[0]+rs[0])/2
        mb = (ls[1]+rs[1])/2

        # dist_btwn nose and midpoint of two shoulders:
        length = math.sqrt(
            ((ma-nose[0])*(ma-nose[0]))+((mb-nose[1])*(mb-nose[1]))
        )
        print(length)

    except:
        pass

    cap = cv2.VideoCapture(0)
