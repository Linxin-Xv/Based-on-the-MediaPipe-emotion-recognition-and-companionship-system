import numpy as np
import cv2
import mediapipe as mp
import pickle
import time

def initialize_model():
    with open('E:/pythonproject/model/mp_model.p', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']

def initialize_mediapipe_facemesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,  # 设为False用于视频流
        max_num_faces=1,          # 只检测一张脸
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame(frame, clf, facemesh, label_classes):
    # 转换颜色空间
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 处理帧
    results = facemesh.process(frame_rgb)
    
    prediction_expression = "Unknown"
    confidence = 0.0
    
    if results.multi_face_landmarks:
        landmark_data = []
        for landmarks in results.multi_face_landmarks:
            # 绘制关键点
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                landmark_data.extend([landmark.x, landmark.y, landmark.z])
                
        # 预测表情
        if landmark_data:
            prediction_index = clf.predict([landmark_data])[0]
            prediction_label = label_classes[prediction_index]
            confidence = np.max(clf.predict_proba([landmark_data]))
            prediction_expression = prediction_label.capitalize()
    
    return frame, prediction_expression, confidence

def main():
    # 初始化模型和FaceMesh
    clf = initialize_model()
    facemesh = initialize_mediapipe_facemesh()
    label_classes = ['happy', 'sad', 'surprise']
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置视频分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 用于控制预测频率的变量
    last_prediction_time = time.time()
    prediction_interval = 0.5  # 每0.5秒预测一次
    current_expression = "Unknown"
    current_confidence = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取画面")
                break
            
            # 翻转画面（如果需要）
            frame = cv2.flip(frame, 1)
            
            # 每隔一定时间进行预测
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                frame, current_expression, current_confidence = process_frame(
                    frame, clf, facemesh, label_classes
                )
                last_prediction_time = current_time
            
            # 在画面上显示预测结果
            text = f"{current_expression} ({current_confidence*100:.2f}%)"
            cv2.putText(
                frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )
            
            # 显示画面
            cv2.imshow('Expression Recognition', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()












# import numpy as np
# import cv2
# import mediapipe as mp
# import pickle
# import time

# def initialize_model():
#     with open('E:/pythonproject/model/mp_model.p', 'rb') as f:
#         model_data = pickle.load(f)
#     return model_data['model']

# def initialize_mediapipe_facemesh():
#     return mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=False,  # 设为False用于视频流
#         max_num_faces=1,          # 只检测一张脸
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

# def process_frame(frame, clf, facemesh, label_classes):
#     # 转换颜色空间
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # 处理帧
#     results = facemesh.process(frame_rgb)
    
#     prediction_expression = "Unknown"
#     confidence = 0.0
    
#     if results.multi_face_landmarks:
#         landmark_data = []
#         for landmarks in results.multi_face_landmarks:
#             # 绘制关键点
#             for landmark in landmarks.landmark:
#                 x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 landmark_data.extend([landmark.x, landmark.y, landmark.z])
                
#         # 预测表情
#         if landmark_data:
#             prediction_index = clf.predict([landmark_data])[0]
#             prediction_label = label_classes[prediction_index]
#             confidence = np.max(clf.predict_proba([landmark_data]))
#             prediction_expression = prediction_label.capitalize()
    
#     return frame, prediction_expression, confidence

# def main():
#     # 初始化模型和FaceMesh
#     clf = initialize_model()
#     facemesh = initialize_mediapipe_facemesh()
#     label_classes = ['happy', 'sad', 'surprise']
    
#     # IP摄像头地址
#     ip_camera_url = "http://192.168.52.107:2056/video"
    
#     # 打开IP摄像头
#     cap = cv2.VideoCapture(ip_camera_url)
#     if not cap.isOpened():
#         print("无法连接到IP摄像头")
#         return
    
#     # 设置视频分辨率
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     # 用于控制预测频率的变量
#     last_prediction_time = time.time()
#     prediction_interval = 0.5  # 每0.5秒预测一次
#     current_expression = "Unknown"
#     current_confidence = 0.0
    
#     print("连接成功！按 'q' 键退出程序")
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("无法获取画面")
#                 # 尝试重新连接
#                 cap = cv2.VideoCapture(ip_camera_url)
#                 continue
            
#             # 计算和显示FPS
#             current_time = time.time()
#             fps = 1.0 / (current_time - last_prediction_time)
            
#             # 每隔一定时间进行预测
#             if current_time - last_prediction_time >= prediction_interval:
#                 frame, current_expression, current_confidence = process_frame(
#                     frame, clf, facemesh, label_classes
#                 )
#                 last_prediction_time = current_time
            
#             # 在画面上显示预测结果和FPS
#             text = f"{current_expression} ({current_confidence*100:.2f}%)"
#             cv2.putText(
#                 frame, text, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 255, 0), 2
#             )
            
#             cv2.putText(
#                 frame, f"FPS: {fps:.2f}", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 255, 0), 2
#             )
            
#             # 显示画面
#             cv2.imshow('Expression Recognition', frame)
            
#             # 按'q'退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     except Exception as e:
#         print(f"发生错误: {e}")
        
#     finally:
#         # 释放资源
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()