import numpy as np
import cv2
import mediapipe as mp
import pickle
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import json
import ollama
from datetime import datetime
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EmotionCompanionSystem:
    def __init__(self):
        self.config = {
            'camera_id': 0,
            'detection_interval': 0.5,
            'model_path': 'E:/pythonproject/model/mp_model.p',
            'ollama_model': 'llama3-chinese',
            'silent_time': 10
        }
        
        self.running = False
        self.should_process_emotion = False
        self.last_response_time = datetime.now()
        self.current_emotion = "Unknown"
        self.current_confidence = 0.0
        self.current_landmarks = None
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.initialize_models()
        self.setup_gui()
        
        self.cap = cv2.VideoCapture(self.config['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def initialize_models(self):
        with open(self.config['model_path'], 'rb') as f:
            model_data = pickle.load(f)
        self.clf = model_data['model']
        self.label_classes = ['happy', 'sad', 'surprise']
        
        self.facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("情感陪伴系统")
        self.root.geometry("850x650")
        
        self.create_gui_elements()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_gui_elements(self):
        # 状态框
        status_frame = ttk.LabelFrame(self.root, text="系统状态", padding="5")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="准备就绪")
        self.status_label.pack()
        
        # 情绪显示框
        emotion_frame = ttk.LabelFrame(self.root, text="当前情绪", padding="5")
        emotion_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.emotion_label = ttk.Label(emotion_frame, text="等待检测...")
        self.emotion_label.pack()
        
        self.detect_button = ttk.Button(emotion_frame, text="开始检测表情", 
                                      command=self.start_emotion_processing)
        self.detect_button.pack(pady=5)
        
        # 对话框
        chat_frame = ttk.LabelFrame(self.root, text="对话", padding="5")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_text = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD)
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        
        # 输入框和发送按钮 
        input_frame = ttk.Frame(self.root, padding="5")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        self.send_button = ttk.Button(input_frame, text="发送", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # 控制按钮
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="启动", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state='disabled')
        
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmark_data = []
            for landmarks in results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    landmark_data.extend([landmark.x, landmark.y, landmark.z])
            
            self.current_landmarks = landmark_data
            
        return frame
        
    def analyze_emotion(self):
        if not self.current_landmarks:
            return "Unknown", 0.0
            
        prediction_index = self.clf.predict([self.current_landmarks])[0]
        prediction_label = self.label_classes[prediction_index]
        confidence = np.max(self.clf.predict_proba([self.current_landmarks]))
        
        if prediction_label == 'happy' and confidence < 0.99:
            prediction_label = 'sad'
        elif prediction_label == 'surprise' and confidence < 0.80:
            prediction_label = 'sad'
        
        return prediction_label.capitalize(), confidence
    
    def stream_ai_response(self, prompt, emotion=None):
        """流式获取AI回应"""
        try:
            system_prompt = """你是一个温暖的AI助手，请根据用户的情绪和输入提供适当的回应。
            保持对话自然、友善，避免过于机械的回答。"""
            
            if emotion:
                system_prompt += f"\n用户当前情绪: {emotion}"
            
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            # 添加时间戳和AI前缀
            self.add_message("AI: ", add_newline=False)
            current_line = ""
            
            for chunk in ollama.chat(
                model=self.config['ollama_model'],
                messages=messages,
                stream=True
            ):
                if chunk and 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    current_line += content
                    self.append_to_chat(content)
                    self.root.update()
            
            # 在完整回复后添加换行        
            self.append_to_chat("\n")
            
        except Exception as e:
            self.add_message(f"AI响应出错: {str(e)}")
            traceback.print_exc()
    
    def append_to_chat(self, text):
        """追加文本到对话框末尾"""
        self.chat_text.insert(tk.END, text)
        self.chat_text.see(tk.END)
        
    def send_message(self):
        """发送用户消息并获取AI回应"""
        message = self.input_entry.get().strip()
        if not message:
            return
            
        # 禁用发送按钮和输入框
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')
        
        # 显示用户消息
        self.add_message(f"你: {message}")
        self.input_entry.delete(0, tk.END)
        
        # 在新线程中获取AI回应
        self.executor.submit(self.stream_ai_response, message, self.current_emotion).add_done_callback(
            lambda future: self.root.after(0, self.enable_input_controls)
        )
        
    def enable_input_controls(self):
        """重新启用输入控件"""
        self.send_button.config(state='normal')
        self.input_entry.config(state='normal')
        self.input_entry.focus()
        
    def start_emotion_processing(self):
        """开始处理表情数据"""
        if self.current_landmarks:
            emotion, confidence = self.analyze_emotion()
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.update_emotion_display(emotion, confidence)
            
            # 在新线程中获取AI开场白
            self.executor.submit(
                self.stream_ai_response,
                f"请基于我当前的{emotion}情绪状态，给出开场白。",
                emotion
            )
        
    def detection_loop(self):
        """情绪检测循环"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)
                
                cv2.imshow('Expression Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
                    
            except Exception as e:
                self.update_status(f"检测错误: {str(e)}")
                traceback.print_exc()
                time.sleep(1)
                
    def update_status(self, message):
        """更新状态显示"""
        self.status_label.config(text=message)
        
    def update_emotion_display(self, emotion, confidence):
        """更新情绪显示"""
        self.emotion_label.config(text=f"情绪: {emotion} (置信度: {confidence:.2f})")
        
    def add_message(self, message, add_newline=True):
        """添加消息到对话框"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = f"{timestamp} - {message}"
        if add_newline:
            text += "\n"
        self.chat_text.insert(tk.END, text)
        self.chat_text.see(tk.END)
        
    def start(self):
        """启动系统"""
        self.running = True
        self.update_status("系统运行中...")
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.start()
        
    def stop(self):
        """停止系统"""
        self.running = False
        self.update_status("系统已停止")
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join()
        
    def on_closing(self):
        """关闭程序"""
        self.stop()
        self.executor.shutdown()
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """运行程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = EmotionCompanionSystem()
    app.run()