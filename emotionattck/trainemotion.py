import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义情绪标签
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 数据集根目录路径 (train 或 test 文件夹路径)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = EMOTIONS
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for emotion in self.classes:
            emotion_path = os.path.join(self.root_dir, emotion)
            if not os.path.isdir(emotion_path):
                continue
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((img_path, self.class_to_idx[emotion]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            # 返回数据集中的第一张图片作为替代
            img_path, label = self.samples[0]
            image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """创建数据加载器"""
    
    # 定义数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建训练集和测试集
    train_dataset = EmotionDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    test_dataset = EmotionDataset(
        os.path.join(data_dir, 'test'),
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/total:.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(EMOTIONS)))
    class_total = list(0. for i in range(len(EMOTIONS)))
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 打印每个类别的准确率
    for i in range(len(EMOTIONS)):
        print(f'Accuracy of {EMOTIONS[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return 100.*correct/total

def main():
    # 配置参数
    data_dir = '..\\dataset'  # 修改为你的数据集路径
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(data_dir, batch_size)
    
    # 创建模型
    model = EmotionNet(num_classes=len(EMOTIONS)).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        
        # 评估
        test_acc = evaluate(model, test_loader)
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_emotion_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')
        print(f'Best Test Acc: {best_acc:.2f}%')
        print('-' * 60)

def predict_image(model, image_path):
    """预测单张图片的情绪"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
        
    return EMOTIONS[predicted.item()]

if __name__ == '__main__':
    main()