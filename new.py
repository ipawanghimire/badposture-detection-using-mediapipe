import cv2
import mediapipe as mp
import numpy as np
import math

count = 0
# calculate length for a good posture
cap = cv2.VideoCapture(0)


while(True):
    ret, frame = cap.read()
    cv2.imshow('img1', frame)

    cv2.putText(frame, "press 'Y' to click the picture", (0, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 400, 0), 2)
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
        gp_length = math.sqrt(
            ((ma-nose[0])*(ma-nose[0]))+((mb-nose[1])*(mb-nose[1]))
        )
        ggp_length = gp_length + 0.1*gp_length
        lgp_length = gp_length - 0.1*gp_length

    except:
        pass

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # extracting data
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
            # print(length)

            # calc angle between nose midpoint and left shoulder
            rad = np.arctan2(nose[1]-mb, nose[0]-ma) - \
                np.arctan2(ls[1]-mb, ls[0]-ma)
            # print(rad)
            angle = np.abs(rad*180/np.pi)
            # print(angle)
            #angle = calc_angle(ls, rs, nose)
            # print(angle)

            if angle > 100 or angle < 80 or length > ggp_length or length < lgp_length:
                count = count+1
                cv2.putText(image, "COUNT", (75, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (5, 5, 0), 2)
                cv2.putText(image, str(count), (75, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)
                # print(count)

            else:
                count = 0

            if count > 20:
                cv2.putText(image, "****************ALERT!****************", (0, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except:
            pass

        # render detection
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


# calculate_angle of shoulders_line and nose to mid point of shoulder line
