import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import time
import cupy as cp
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_dict_to_arrays(data_dict):
    data = []
    labels = []
    for img_path, landmarks in data_dict.items():
        data.append(np.array(landmarks).flatten())
        labels.append(img_path.split("/")[-2])
    return np.asarray(data), np.asarray(labels)

def split_data(data, labels, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, shuffle=True, stratify=labels
    )
    return x_train, x_test, y_train, y_test

def tune_hyperparameters(x_train, y_train):
    n_classes = len(np.unique(y_train))
    print(f"类别数量: {n_classes}")
    
    # 定义参数网格
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        # 'min_child_weight': [1, 3, 5]
        # 移除gpu相关参数，因为这些会在基础模型中设置
    }

    # 创建基础模型
    base_model = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='gpu_hist',  # 使用GPU
        predictor='gpu_predictor',  # GPU预测
        gpu_id=0,  # 使用第一个GPU
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    # 创建网格搜索
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=1,
        verbose=2,
        scoring='accuracy'
    )
    
    print("\n开始网格搜索...")
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n训练完成！用时: {training_time:.2f} 秒")
    print("最佳参数:", grid_search.best_params_)
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳参数更新基础模型
    best_params = grid_search.best_params_
    best_model = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        **best_params  # 添加最佳参数
    )
    
    # 训练最终模型
    print("\n训练最终模型...")
    best_model.fit(x_train, y_train)
    
    return best_model

def evaluate_model(model, x_test, y_test):
    print("\n开始模型评估...")
    start_time = time.time()
    y_predict = model.predict(x_test)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_predict)
    report = classification_report(y_test, y_predict)
    matrix = confusion_matrix(y_test, y_predict)
    
    print(f"预测完成！用时: {prediction_time:.2f} 秒")
    return accuracy, report, matrix

def plot_confusion_matrix(matrix, labels, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_metrics(matrix, labels, accuracy, report):
    plt.figure(figsize=(10,5))
    plot_confusion_matrix(matrix, labels)
    plt.show()
    print(f"\nClassification Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n", report)

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

def check_gpu():
    """检查GPU可用性"""
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"\n发现 {n_devices} 个 GPU 设备:")
        
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"GPU {i}: {props['name'].decode()}")
            print(f"总内存: {props['totalGlobalMem'] / (1024**3):.2f} GB")
            print(f"计算能力: {props['major']}.{props['minor']}")
            print(f"最大线程数/块: {props['maxThreadsPerBlock']}")
            print("-" * 50)
        
        return True
    except Exception as e:
        print(f"\nGPU检查失败: {str(e)}")
        print("请确保已正确安装CUDA和cupy")
        return False

def main():
    try:
        # 检查GPU
        if not check_gpu():
            return

        print("\n开始加载数据...")
        data_dict = load_data('E:/pythonproject/mp_landmark_data.pkl')
        
        print("转换数据格式...")
        data, labels = convert_dict_to_arrays(data_dict)
        
        print("编码标签...")
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        label_classes = label_encoder.classes_
        
        print(f"\n数据统计:")
        print(f"样本数量: {data.shape[0]}")
        print(f"特征维度: {data.shape[1]}")
        print(f"类别数量: {len(label_classes)}")
        print(f"类别分布: {np.unique(labels, return_counts=True)[1]}")
        
        print("\n拆分训练集和测试集...")
        x_train, x_test, y_train, y_test = split_data(data, labels_encoded)
        print(f"训练集大小: {x_train.shape}")
        print(f"测试集大小: {x_test.shape}")

        print("\n开始模型训练和优化...")
        best_model = tune_hyperparameters(x_train, y_train)
        
        print("\n评估模型性能...")
        accuracy, report, matrix = evaluate_model(best_model, x_test, y_test)
        
        print("\n绘制评估指标...")
        plot_metrics(matrix, label_classes, accuracy, report)
        
        print("\n保存模型...")
        save_model(best_model, 'E:/pythonproject/model/mp_model_gpu.pkl')
        print("模型已保存至 'E:/pythonproject/model/mp_model_gpu.pkl'")
        
        print("\n处理完成!")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()