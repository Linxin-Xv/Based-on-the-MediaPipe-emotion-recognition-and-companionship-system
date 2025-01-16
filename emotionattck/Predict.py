import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from tqdm import tqdm


class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EmotionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(EmotionNN, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EmotionNNPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': []}

        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def prepare_data(self, csv_path, batch_size=32):
        try:
            print("正在加载数据...")
            df = pd.read_csv(csv_path)

            X = df[['eyebrow_height', 'mouth_aspect_ratio', 'eye_aspect_ratio',
                    'nose_height', 'eyebrow_distance']].values
            y = df['emotion'].values

            y_encoded = self.label_encoder.fit_transform(y)
            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            train_dataset = EmotionDataset(X_train, y_train)
            test_dataset = EmotionDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            print(f"训练集大小: {len(X_train)}")
            print(f"测试集大小: {len(X_test)}")
            print(f"特征维度: {X_train.shape[1]}")
            print(f"类别数量: {len(self.label_encoder.classes_)}")
            print("类别分布:")
            for label, count in zip(self.label_encoder.classes_,
                                    np.bincount(y_encoded)):
                print(f"{label}: {count}")

            return train_loader, test_loader, X_train.shape[1], \
                len(self.label_encoder.classes_)

        except Exception as e:
            print(f"数据准备错误: {str(e)}")
            return None, None, None, None

    def train(self, train_loader, test_loader, input_size, num_classes,
              hidden_sizes=[128, 64, 32], num_epochs=100, learning_rate=0.001,
              early_stopping_patience=10):
        try:
            # 创建保存目录
            save_dir = 'saved_models'
            os.makedirs(save_dir, exist_ok=True)

            self.model = EmotionNN(input_size, hidden_sizes, num_classes)
            self.model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                             patience=5)

            best_val_loss = float('inf')
            patience_counter = 0

            print("\n开始训练...")
            for epoch in range(num_epochs):
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
                for inputs, labels in train_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * train_correct / train_total:.2f}%'
                    })

                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(test_loader)
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total

                scheduler.step(avg_val_loss)

                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(avg_val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)

                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'history': self.history
                    }, model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\n早停: {early_stopping_patience} 个epoch未改善")
                        break

            # 加载最佳模型
            model_path = os.path.join(save_dir, 'best_model.pth')
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.plot_training_history()

            return True

        except Exception as e:
            print(f"训练错误: {str(e)}")
            return False

    def evaluate(self, test_loader):
        try:
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            print("\n模型评估报告:")
            print(classification_report(all_labels, all_preds,
                                        target_names=self.label_encoder.classes_))

            self.plot_confusion_matrix(all_labels, all_preds)

            return all_labels, all_preds

        except Exception as e:
            print(f"评估错误: {str(e)}")
            return None, None

    def plot_training_history(self):
        try:
            plots_dir = 'training_plots'
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Val Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_acc'], label='Train Acc')
            plt.plot(self.history['val_acc'], label='Val Acc')
            plt.title('Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'training_history.png'))
            plt.close()

        except Exception as e:
            print(f"绘图错误: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred):
        try:
            plots_dir = 'training_plots'
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix_nn.png'))
            plt.close()

        except Exception as e:
            print(f"绘图错误: {str(e)}")

    def save_model(self, model_path='emotion_model_nn'):
        try:
            save_dir = 'saved_models'
            os.makedirs(save_dir, exist_ok=True)

            model_full_path = os.path.join(save_dir, f'{model_path}.pth')
            scaler_path = os.path.join(save_dir, 'scaler.joblib')
            encoder_path = os.path.join(save_dir, 'label_encoder.joblib')

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history
            }, model_full_path)

            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)

            print(f"模型和相关文件已保存到 {save_dir} 目录")
            return True

        except Exception as e:
            print(f"保存错误: {str(e)}")
            return False

    def load_model(self, model_path='emotion_model_nn'):
        try:
            save_dir = 'saved_models'
            model_full_path = os.path.join(save_dir, f'{model_path}.pth')
            scaler_path = os.path.join(save_dir, 'scaler.joblib')
            encoder_path = os.path.join(save_dir, 'label_encoder.joblib')

            checkpoint = torch.load(model_full_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint['history']

            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)

            print("模型和相关文件加载成功")
            return True

        except Exception as e:
            print(f"加载错误: {str(e)}")
            return False

    def predict(self, features):
        try:
            self.model.eval()

            if isinstance(features, pd.DataFrame):
                features = features.values
            if features.ndim == 1:
                features = features.reshape(1, -1)

            features_scaled = self.scaler.transform(features)
            inputs = torch.FloatTensor(features_scaled).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

            predictions = self.label_encoder.inverse_transform(
                predicted.cpu().numpy()
            )

            return predictions, probabilities.cpu().numpy()

        except Exception as e:
            print(f"预测错误: {str(e)}")
            return None, None


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    total_start_time = time.time()

    predictor = EmotionNNPredictor()

    csv_path = 'train_features.csv'  # 修改为你的CSV文件路径
    train_loader, test_loader, input_size, num_classes = \
        predictor.prepare_data(csv_path, batch_size=32)

    if train_loader is None:
        return

    if predictor.train(train_loader, test_loader, input_size, num_classes,
                       hidden_sizes=[128, 64, 32], num_epochs=100,
                       learning_rate=0.001, early_stopping_patience=10):
        predictor.evaluate(test_loader)
        predictor.save_model()

        test_inputs, _ = next(iter(test_loader))
        test_features = test_inputs[0].numpy()

        predictions, probabilities = predictor.predict(test_features)

        if predictions is not None:
            print("\n测试预测:")
            print(f"预测的情绪: {predictions[0]}")
            print("\n各类别概率:")
            for emotion, prob in zip(predictor.label_encoder.classes_,
                                     probabilities[0]):
                print(f"{emotion}: {prob:.4f}")

    total_end_time = time.time()
    print(f"\n总用时: {total_end_time - total_start_time:.2f} 秒")


if __name__ == "__main__":
    main()