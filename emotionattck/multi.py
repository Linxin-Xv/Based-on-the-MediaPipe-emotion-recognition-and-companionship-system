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
from concurrent.futures import ThreadPoolExecutor

class CustomStyle:
    # 颜色主题
    COLORS = {
        'bg_primary': '#f0f2f5',           # 主背景色
        'bg_secondary': '#ffffff',          # 次背景色
        'text_primary': '#1a1a1a',          # 主文本色
        'text_secondary': '#666666',        # 次文本色
        'accent': '#1890ff',               # 强调色
        'accent_hover': '#40a9ff',         # 强调色悬停
        'success': '#52c41a',              # 成功色
        'warning': '#faad14',              # 警告色
        'error': '#f5222d',                # 错误色
        'border': '#d9d9d9',               # 边框色
        'chat_user_bg': '#e6f7ff',         # 用户消息背景
        'chat_ai_bg': '#f6ffed',           # AI消息背景
    }
    
    # 字体设置
    FONTS = {
        'main': ('Microsoft YaHei UI', 10),
        'title': ('Microsoft YaHei UI', 12, 'bold'),
        'chat': ('Microsoft YaHei UI', 10),
        'button': ('Microsoft YaHei UI', 9),
    }

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
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.initialize_models()
        self.setup_gui()
        
        self.cap = cv2.VideoCapture(self.config['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
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
        self.root.geometry("900x700")
        self.root.configure(bg=CustomStyle.COLORS['bg_primary'])
        
        # 设置ttk样式
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 使用clam主题作为基础
        
        # 配置自定义样式
        self.configure_styles()
        
        self.create_gui_elements()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def configure_styles(self):
        # 配置标签框样式
        self.style.configure(
            'Custom.TLabelframe',
            background=CustomStyle.COLORS['bg_secondary'],
            borderwidth=1,
            relief='solid'
        )
        self.style.configure(
            'Custom.TLabelframe.Label',
            background=CustomStyle.COLORS['bg_secondary'],
            foreground=CustomStyle.COLORS['text_primary'],
            font=CustomStyle.FONTS['title']
        )
        
        # 配置按钮样式
        self.style.configure(
            'Accent.TButton',
            background=CustomStyle.COLORS['accent'],
            foreground='white',
            font=CustomStyle.FONTS['button'],
            padding=(10, 5)
        )
        self.style.map(
            'Accent.TButton',
            background=[('active', CustomStyle.COLORS['accent_hover'])]
        )
        
        # 配置普通按钮样式
        self.style.configure(
            'Normal.TButton',
            font=CustomStyle.FONTS['button'],
            padding=(10, 5)
        )
        
        # 配置标签样式
        self.style.configure(
            'Custom.TLabel',
            background=CustomStyle.COLORS['bg_secondary'],
            foreground=CustomStyle.COLORS['text_primary'],
            font=CustomStyle.FONTS['main']
        )
        
        # 配置状态标签样式
        self.style.configure(
            'Status.TLabel',
            background=CustomStyle.COLORS['bg_secondary'],
            foreground=CustomStyle.COLORS['text_secondary'],
            font=CustomStyle.FONTS['main']
        )
    def create_gui_elements(self):
        # 创建主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 顶部状态区域
        top_frame = ttk.LabelFrame(
            main_container,
            text="系统状态",
            style='Custom.TLabelframe'
        )
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        status_container = ttk.Frame(top_frame)
        status_container.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(
            status_container,
            text="准备就绪",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT)
        
        # 情绪显示区域
        emotion_frame = ttk.LabelFrame(
            main_container,
            text="情绪状态",
            style='Custom.TLabelframe'
        )
        emotion_frame.pack(fill=tk.X, pady=(0, 10))
        
        emotion_container = ttk.Frame(emotion_frame)
        emotion_container.pack(fill=tk.X, padx=10, pady=5)
        
        self.emotion_label = ttk.Label(
            emotion_container,
            text="等待检测...",
            style='Custom.TLabel'
        )
        self.emotion_label.pack(side=tk.LEFT)
        
        self.detect_button = ttk.Button(
            emotion_container,
            text="开始检测表情",
            command=self.start_emotion_processing,
            style='Accent.TButton'
        )
        self.detect_button.pack(side=tk.RIGHT)
        
        # 对话区域
        chat_frame = ttk.LabelFrame(
            main_container,
            text="对话",
            style='Custom.TLabelframe'
        )
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 自定义聊天框样式
        chat_style = {
            'font': CustomStyle.FONTS['chat'],
            'bg': CustomStyle.COLORS['bg_secondary'],
            'fg': CustomStyle.COLORS['text_primary'],
            'selectbackground': CustomStyle.COLORS['accent'],
            'insertbackground': CustomStyle.COLORS['text_primary'],
            'relief': 'flat',
            'padx': 5,
            'pady': 5
        }
        
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            **chat_style
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 输入区域
        input_frame = ttk.Frame(main_container)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_entry = ttk.Entry(
            input_frame,
            font=CustomStyle.FONTS['main']
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        self.send_button = ttk.Button(
            input_frame,
            text="发送",
            command=self.send_message,
            style='Accent.TButton'
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(
            control_frame,
            text="启动系统",
            command=self.start,
            style='Normal.TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            control_frame,
            text="停止系统",
            command=self.stop,
            style='Normal.TButton'
        )
        self.stop_button.pack(side=tk.LEFT)
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
        
        if prediction_label == 'happy' and confidence < 0.90:
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
            self.update_status("AI响应出错", 'error')
            traceback.print_exc()
    
    def format_chat_message(self, speaker, message):
        """格式化聊天消息，为不同发送者添加不同样式"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if speaker == "你":
            self.chat_text.tag_config(
                "user_msg",
                background=CustomStyle.COLORS['chat_user_bg'],
                lmargin1=20,
                lmargin2=20
            )
            self.chat_text.insert(tk.END, f"{timestamp} - {speaker}: ", "time_stamp")
            self.chat_text.insert(tk.END, f"{message}\n", "user_msg")
        else:
            self.chat_text.tag_config(
                "ai_msg",
                background=CustomStyle.COLORS['chat_ai_bg'],
                lmargin1=20,
                lmargin2=20
            )
            self.chat_text.insert(tk.END, f"{timestamp} - {speaker}: ", "time_stamp")
            self.chat_text.insert(tk.END, f"{message}\n", "ai_msg")
        
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
        self.format_chat_message("你", message)
        self.input_entry.delete(0, tk.END)
        
        # 在新线程中获取AI回应
        self.executor.submit(
            self.stream_ai_response,
            message,
            self.current_emotion
        ).add_done_callback(
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
                self.update_status(f"检测错误: {str(e)}", 'error')
                traceback.print_exc()
                time.sleep(1)
    
    def update_status(self, message, status_type='info'):
        """使用不同颜色显示不同类型的状态消息"""
        color = {
            'info': CustomStyle.COLORS['text_secondary'],
            'success': CustomStyle.COLORS['success'],
            'warning': CustomStyle.COLORS['warning'],
            'error': CustomStyle.COLORS['error']
        }.get(status_type, CustomStyle.COLORS['text_secondary'])
        
        self.status_label.configure(foreground=color)
        self.status_label.configure(text=message)
        
    def update_emotion_display(self, emotion, confidence):
        """更新情绪显示"""
        text = f"当前情绪: {emotion} (置信度: {confidence:.2f})"
        color = CustomStyle.COLORS['success'] if confidence > 0.7 else CustomStyle.COLORS['warning']
        self.emotion_label.configure(foreground=color)
        self.emotion_label.configure(text=text)

    def add_message(self, message, add_newline=True):
        """添加消息到对话框"""
        if message.startswith("你: "):
            self.format_chat_message("你", message[3:])
        elif message.startswith("AI: "):
            self.format_chat_message("AI", message[4:])
        else:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.chat_text.insert(tk.END, f"{timestamp} - {message}")
            if add_newline:
                self.chat_text.insert(tk.END, "\n")
        self.chat_text.see(tk.END)

    def append_to_chat(self, text):
        """追加文本到对话框末尾"""
        self.chat_text.insert(tk.END, text)
        self.chat_text.see(tk.END)
                
    def start(self):
        """启动系统"""
        self.running = True
        self.update_status("系统运行中...", 'success')
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
    