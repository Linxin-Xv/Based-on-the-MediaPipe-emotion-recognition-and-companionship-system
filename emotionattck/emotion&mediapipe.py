import mediapipe as mp
import cv2
import numpy as np
import tkinter as tk
from collections import deque
import time

class EmotionWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.title("Real-time Emotion")
        self.attributes('-topmost', True)
        self.attributes('-alpha', 0.8)
        self.configure(bg='black')
        
        # 创建情绪显示标签
        self.emotion_label = tk.Label(
            self,
            text="Detecting...",
            font=("Arial", 16, "bold"),
            bg="black",
            fg="white",
            padx=20,
            pady=10
        )
        self.emotion_label.pack()
        
        # 创建情绪值显示条
        self.canvas = tk.Canvas(
            self, 
            width=200, 
            height=20, 
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack(pady=5)
        
        # 创建情绪值进度条
        self.progress_bar = self.canvas.create_rectangle(
            0, 0, 0, 20,
            fill="green",
            width=0
        )
        
        # 设置窗口位置
        self.geometry(f"+{self.winfo_screenwidth()-300}+50")
        
    def update_display(self, emotion, intensity, color):
        """更新情绪显示"""
        # 更新文本
        self.emotion_label.config(
            text=f"Current Emotion:\n{emotion}",
            fg=color
        )
        
        # 更新进度条
        bar_width = min(200, max(0, intensity * 200))
        self.canvas.coords(self.progress_bar, 0, 0, bar_width, 20)
        self.canvas.itemconfig(self.progress_bar, fill=color)

class DynamicEmotionDetector:
    def __init__(self):
        # 初始化MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 面部特征检测器
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 情绪相关参数
        self.emotion_window = deque(maxlen=15)  # 情绪历史窗口
        self.intensity_window = deque(maxlen=15)  # 强度历史窗口
        self.last_time = time.time()
        
        # 定义面部关键点索引
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        self.MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # 创建显示窗口
        self.emotion_window_ui = EmotionWindow()
        
        # 情绪颜色映射
        self.emotion_colors = {
            "Happy": "#00FF00",     # 绿色
            "Sad": "#4169E1",       # 蓝色
            "Angry": "#FF0000",     # 红色
            "Surprised": "#FFD700",  # 金色
            "Neutral": "#FFFFFF",    # 白色
            "Contempt": "#800080",   # 紫色
            "Disgusted": "#8B4513"   # 棕色
        }

    def calculate_facial_features(self, landmarks):
        """计算面部特征参数"""
        # 转换landmarks为numpy数组
        points = np.array([[l.x, l.y, l.z] for l in landmarks])
        
        # 计算眉毛特征
        left_eyebrow_height = np.mean([points[i][1] for i in self.LEFT_EYEBROW])
        right_eyebrow_height = np.mean([points[i][1] for i in self.RIGHT_EYEBROW])
        eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # 计算嘴部特征
        mouth_points = np.array([points[i] for i in self.MOUTH_OUTLINE])
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
        mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
        mouth_aspect_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # 计算眼睛开合度
        left_eye_points = np.array([points[i] for i in self.LEFT_EYE])
        right_eye_points = np.array([points[i] for i in self.RIGHT_EYE])
        left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
        right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
        eye_aspect_ratio = (left_eye_height + right_eye_height) / 2
        
        return eyebrow_height, mouth_aspect_ratio, eye_aspect_ratio

    def detect_emotion(self, landmarks):
        """检测情绪状态"""
        eyebrow_height, mouth_ratio, eye_ratio = self.calculate_facial_features(landmarks)
        
        # 动态情绪判断逻辑
        intensity = 0.0
        emotion = "Neutral"
        
        # 快乐/微笑
        if mouth_ratio > 2.5:
            emotion = "Happy"
            intensity = min((mouth_ratio - 2.5) / 2.0, 1.0)
        
        # 悲伤
        elif mouth_ratio < 2.8 and eyebrow_height > 0.33:
            emotion = "Sad"
            intensity = min((0.33 - eyebrow_height) * 3, 1.0)
        
        # 愤怒
        elif eyebrow_height < 0.28:
            emotion = "Angry"
            intensity = min((0.28 - eyebrow_height) * 5, 1.0)
        
        # 惊讶
        elif eye_ratio > 0.15:
            emotion = "Surprised"
            intensity = min((eye_ratio - 0.15) * 10, 1.0)
        
        # 中性
        else:
            intensity = 0.5
        
        return emotion, intensity

    def draw_debug_info(self, image, landmarks):
        """绘制调试信息"""
        for landmark in landmarks:
            pos = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
            cv2.circle(image, pos, 1, (0, 255, 0), -1)

    def run(self):
        """主循环"""
        cap = cv2.VideoCapture(0)
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                # 转换为RGB处理
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 检测情绪
                        emotion, intensity = self.detect_emotion(face_landmarks.landmark)
                        
                        # 添加到历史窗口
                        self.emotion_window.append(emotion)
                        self.intensity_window.append(intensity)
                        
                        # 获取主导情绪和平均强度
                        if len(self.emotion_window) == self.emotion_window.maxlen:
                            dominant_emotion = max(set(self.emotion_window), 
                                                key=self.emotion_window.count)
                            avg_intensity = sum(self.intensity_window) / len(self.intensity_window)
                            
                            # 更新显示
                            color = self.emotion_colors.get(dominant_emotion, "#FFFFFF")
                            self.emotion_window_ui.update_display(
                                dominant_emotion,
                                avg_intensity,
                                color
                            )
                        
                        # 绘制调试信息
                        self.draw_debug_info(image, face_landmarks.landmark)
                        
                        # 显示情绪文本
                        cv2.putText(
                            image,
                            f"{emotion} ({intensity:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                # 显示图像
                cv2.imshow('Dynamic Emotion Detection', image)
                
                # 更新UI
                self.emotion_window_ui.update()
                
                # 检查退出
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.emotion_window_ui.destroy()

if __name__ == "__main__":
    detector = DynamicEmotionDetector()
    detector.run()




# import mediapipe as mp
# import cv2
# import numpy as np
# import tkinter as tk
# from fer import FER
# from collections import deque
# import time

# class EmotionWindow(tk.Tk):
#     def __init__(self):
#         super().__init__()
        
#         # 窗口设置
#         self.title("Emotion Detection")
#         self.attributes('-topmost', True)
#         self.attributes('-alpha', 0.8)
#         self.configure(bg='black')
        
#         # 主情绪显示
#         self.emotion_label = tk.Label(
#             self,
#             text="Analyzing...",
#             font=("Arial", 16, "bold"),
#             bg="black",
#             fg="white",
#             padx=20,
#             pady=10
#         )
#         self.emotion_label.pack()
        
#         # 情绪概率条框架
#         self.bars_frame = tk.Frame(self, bg='black')
#         self.bars_frame.pack(pady=10, padx=20)
        
#         # 初始化情绪条
#         self.emotion_bars = {}
#         self.emotion_labels = {}
#         emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#         colors = ['#FF0000', '#800080', '#4B0082', '#00FF00', '#0000FF', '#FFD700', '#FFFFFF']
        
#         for emotion, color in zip(emotions, colors):
#             frame = tk.Frame(self.bars_frame, bg='black')
#             frame.pack(fill='x', pady=2)
            
#             # 情绪标签
#             label = tk.Label(
#                 frame,
#                 text=f"{emotion.capitalize()}: 0%",
#                 font=("Arial", 10),
#                 bg='black',
#                 fg=color,
#                 width=15,
#                 anchor='w'
#             )
#             label.pack(side='left')
            
#             # 情绪条
#             canvas = tk.Canvas(
#                 frame,
#                 width=150,
#                 height=10,
#                 bg='black',
#                 highlightthickness=0
#             )
#             canvas.pack(side='left', padx=5)
            
#             # 创建进度条
#             bar = canvas.create_rectangle(0, 0, 0, 10, fill=color, width=0)
            
#             self.emotion_bars[emotion] = (canvas, bar)
#             self.emotion_labels[emotion] = label
        
#         # 设置窗口位置
#         self.geometry(f"+{self.winfo_screenwidth()-300}+50")
    
#     def update_emotions(self, emotions_dict):
#         """更新所有情绪概率显示"""
#         # 找出最强情绪
#         dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
        
#         # 更新主情绪显示
#         self.emotion_label.config(
#             text=f"Current Emotion:\n{dominant_emotion[0].capitalize()}",
#             fg=self._get_emotion_color(dominant_emotion[0])
#         )
        
#         # 更新所有情绪条
#         for emotion, probability in emotions_dict.items():
#             if emotion in self.emotion_bars:
#                 canvas, bar = self.emotion_bars[emotion]
#                 width = int(probability * 150)  # 转换概率为宽度
#                 canvas.coords(bar, 0, 0, width, 10)
                
#                 # 更新标签
#                 self.emotion_labels[emotion].config(
#                     text=f"{emotion.capitalize()}: {probability*100:.1f}%"
#                 )
    
#     @staticmethod
#     def _get_emotion_color(emotion):
#         """返回情绪对应的颜色"""
#         colors = {
#             'angry': '#FF0000',
#             'disgust': '#800080',
#             'fear': '#4B0082',
#             'happy': '#00FF00',
#             'sad': '#0000FF',
#             'surprise': '#FFD700',
#             'neutral': '#FFFFFF'
#         }
#         return colors.get(emotion, '#FFFFFF')

# class EmotionDetector:
#     def __init__(self):
#         # 初始化MediaPipe Face Detection
#         self.mp_face_detection = mp.solutions.face_detection
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.face_detection = self.mp_face_detection.FaceDetection(
#             model_selection=0,
#             min_detection_confidence=0.5
#         )
        
#         # 初始化FER模型
#         self.emotion_detector = FER(mtcnn=True)
        
#         # 创建显示窗口
#         self.emotion_window = EmotionWindow()
        
#         # 初始化情绪历史队列
#         self.emotion_history = deque(maxlen=10)
        
#         # FPS计算
#         self.fps_time = time.time()
#         self.fps = 0
        
#     def process_frame(self, frame):
#         """处理单帧图像"""
#         # 转换颜色空间
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # 检测情绪
#         emotions = self.emotion_detector.detect_emotions(frame)
        
#         if emotions:
#             # 获取情绪概率
#             emotions_dict = emotions[0]['emotions']
#             self.emotion_history.append(emotions_dict)
            
#             # 计算平均情绪概率
#             if len(self.emotion_history) == self.emotion_history.maxlen:
#                 avg_emotions = {}
#                 for emotion in emotions_dict.keys():
#                     avg_emotions[emotion] = np.mean([h[emotion] for h in self.emotion_history])
                
#                 # 更新显示
#                 self.emotion_window.update_emotions(avg_emotions)
            
#             # 在图像上绘制边界框和标签
#             face = emotions[0]['box']
#             x, y, w, h = [int(v) for v in face]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             # 显示主要情绪
#             dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
#             cv2.putText(
#                 frame,
#                 f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}",
#                 (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (255, 0, 0),
#                 2
#             )
        
#         # 计算并显示FPS
#         current_time = time.time()
#         self.fps = 1 / (current_time - self.fps_time)
#         self.fps_time = current_time
        
#         cv2.putText(
#             frame,
#             f"FPS: {self.fps:.1f}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )
        
#         return frame

#     def run(self):
#         """主循环"""
#         cap = cv2.VideoCapture(0)
        
#         try:
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     continue
                
#                 # 处理帧
#                 processed_frame = self.process_frame(frame)
                
#                 # 显示结果
#                 cv2.imshow('Emotion Detection', processed_frame)
                
#                 # 更新UI
#                 self.emotion_window.update()
                
#                 # 检查退出
#                 if cv2.waitKey(1) & 0xFF == 27:
#                     break
                    
#         finally:
#             cap.release()
#             cv2.destroyAllWindows()
#             self.emotion_window.destroy()

# if __name__ == "__main__":
#     detector = EmotionDetector()
#     detector.run()