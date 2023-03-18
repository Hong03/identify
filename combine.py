import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils            # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles    # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands                      # mediapipe 偵測手掌方法
mp_pose = mp.solutions.pose                        # mediapipe 姿勢偵測

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:

    with mp_hands.Hands(
        model_complexity = 0,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as hands:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img, (1024,600))                 # 視窗大小
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # 辨識的顏色
            results_pose = pose.process(img2)                 # 姿體偵測
            results_hands = hands.process(img2)               # 手掌偵測
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # 辨識手掌的結果，並標記
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    # 辨識姿體的結果，並標記
                    mp_drawing.draw_landmarks(
                        img,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style()
                    )

            cv2.imshow('media', img)
            if cv2.waitKey(5) == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()


