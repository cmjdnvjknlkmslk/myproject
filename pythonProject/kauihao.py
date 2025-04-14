import cv2
import mediapipe as mp

def detect_hands_in_video(video_path):
    # 初始化MediaPipe Hands模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测手部
        results = hands.process(frame_rgb)

        # 在帧上绘制检测到的手部标记
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # 将关键点的坐标转换为图像坐标
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # 在图像上绘制关键点
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    # 将关键点的坐标打印出来
                    print(f"Hand {idx + 1} - X: {cx}, Y: {cy}")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 显示结果
        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 指定要检测手部的视频文件路径
video_path = "D:\\videosave\\Video_20240417095331366.avi"
detect_hands_in_video(video_path)
