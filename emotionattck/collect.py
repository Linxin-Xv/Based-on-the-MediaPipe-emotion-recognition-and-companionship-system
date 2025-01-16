import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import warnings
import gc
import time

# 抑制警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class FacialFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        # 定义特征点索引
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.NOSE = [1, 2, 98, 327]

    def initialize_face_mesh(self):
        """初始化face_mesh对象"""
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )

    def release_resources(self):
        """释放资源"""
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        gc.collect()

    def extract_features(self, image_path):
        """提取单张图片的特征"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 计算特征
            eyebrow_height, mouth_aspect_ratio, eye_aspect_ratio = self.calculate_facial_features(landmarks)
            
            # 提取更多特征
            nose_tip_pos = landmarks[self.NOSE[0]]
            nose_height = nose_tip_pos.y
            
            # 计算眉间距离
            eyebrow_distance = abs(
                np.mean([landmarks[i].x for i in self.LEFT_EYEBROW]) - 
                np.mean([landmarks[i].x for i in self.RIGHT_EYEBROW])
            )
            
            return [
                eyebrow_height,
                mouth_aspect_ratio,
                eye_aspect_ratio,
                nose_height,
                eyebrow_distance
            ]
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            return None
        finally:
            # 清理内存
            image = None
            results = None
            gc.collect()

    def calculate_facial_features(self, landmarks):
        """计算面部特征参数"""
        points = np.array([[l.x, l.y, l.z] for l in landmarks])
        
        # 眉毛高度
        left_eyebrow_height = np.mean([points[i][1] for i in self.LEFT_EYEBROW])
        right_eyebrow_height = np.mean([points[i][1] for i in self.RIGHT_EYEBROW])
        eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # 嘴部特征
        mouth_points = np.array([points[i] for i in self.MOUTH_OUTLINE])
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
        mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
        mouth_aspect_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # 眼睛开合度
        left_eye_points = np.array([points[i] for i in self.LEFT_EYE])
        right_eye_points = np.array([points[i] for i in self.RIGHT_EYE])
        left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
        right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
        eye_aspect_ratio = (left_eye_height + right_eye_height) / 2
        
        return eyebrow_height, mouth_aspect_ratio, eye_aspect_ratio

def process_image_batch(extractor, image_files, emotion_path, emotion, feature_names):
    """处理一批图片"""
    data = []
    processed_count = 0
    failed_count = 0
    
    for image_file in tqdm(image_files, desc=f"Processing {emotion}"):
        try:
            image_path = os.path.join(emotion_path, image_file)
            extractor.initialize_face_mesh()
            features = extractor.extract_features(image_path)
            
            if features is not None:
                features.append(emotion)
                data.append(features)
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            print(f"\nError processing {image_file}: {str(e)}")
            continue
            
        finally:
            extractor.release_resources()
            time.sleep(0.1)  # 每张图片处理后短暂暂停
            
    return data, processed_count, failed_count

def process_dataset_in_batches(data_dir, output_file, batch_size=50):
    """分批处理数据集"""
    extractor = FacialFeatureExtractor()
    feature_names = ['eyebrow_height', 'mouth_aspect_ratio', 'eye_aspect_ratio', 
                    'nose_height', 'eyebrow_distance', 'emotion']
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    total_processed = 0
    total_failed = 0
    
    # 创建或清空输出文件
    pd.DataFrame(columns=feature_names).to_csv(output_file, index=False)
    
    try:
        for emotion in emotions:
            emotion_path = os.path.join(data_dir, emotion)
            if not os.path.exists(emotion_path):
                print(f"Skipping {emotion} - directory not found")
                continue
                
            image_files = [f for f in os.listdir(emotion_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\nProcessing {emotion} images ({len(image_files)} files found)")
            
            # 分批处理图片
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                print(f"\nProcessing batch {i//batch_size + 1} of {len(image_files)//batch_size + 1}")
                
                # 处理当前批次
                data, processed_count, failed_count = process_image_batch(
                    extractor, batch_files, emotion_path, emotion, feature_names
                )
                
                total_processed += processed_count
                total_failed += failed_count
                
                # 保存当前批次的数据
                if data:
                    df = pd.DataFrame(data, columns=feature_names)
                    df.to_csv(output_file, mode='a', header=False, index=False)
                
                print(f"Batch completed. Batch processed: {processed_count}, Batch failed: {failed_count}")
                print(f"Total processed: {total_processed}, Total failed: {total_failed}")
                time.sleep(2)  # 批次间暂停
                
            print(f"\nCompleted processing {emotion}")
            time.sleep(5)  # 情绪类别间暂停
            
    finally:
        extractor.release_resources()
    
    print("\nFinal Processing Summary:")
    print(f"Total successfully processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    
    # 读取并返回完整的数据集
    try:
        return pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading final CSV file: {str(e)}")
        return None

def create_output_directory():
    """创建输出目录"""
    output_dir = 'extracted_features'
    try:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        # 创建输出目录
        output_dir = create_output_directory()
        if output_dir is None:
            raise Exception("Failed to create output directory")
        
        # 处理训练集
        print("Processing training dataset...")
        train_output = os.path.join(output_dir, 'train_features.csv')
        train_df = process_dataset_in_batches(
            data_dir='.\\dataset\\train\\',
            output_file=train_output,
            batch_size=50
        )
        
        if train_df is not None:
            print("\nTrain Dataset Sample:")
            print(train_df.head())
            print("\nTrain Dataset Info:")
            print(train_df.info())
        
        # 处理测试集
        print("\nProcessing test dataset...")
        test_output = os.path.join(output_dir, 'test_features.csv')
        test_df = process_dataset_in_batches(
            data_dir='.\\dataset\\test\\',
            output_file=test_output,
            batch_size=50
        )
        
        if test_df is not None:
            print("\nTest Dataset Sample:")
            print(test_df.head())
            print("\nTest Dataset Info:")
            print(test_df.info())
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcess completed")