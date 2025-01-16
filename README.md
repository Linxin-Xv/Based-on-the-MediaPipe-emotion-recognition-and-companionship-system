# 表情识别与情感陪伴系统

这是一个基于MediaPipe和大语言模型的实时表情识别与情感陪伴系统。系统能够识别用户的表情，并通过本地部署的大语言模型提供情感交互支持。

## 功能特点

- 实时表情识别和追踪
- 基于表情的情感交互对话
- 直观的图形用户界面
- 本地化部署，保护隐私
- GPU加速支持

## 系统要求

- Python 3.10.12
- CUDA支持的NVIDIA GPU (推荐)
- 设备摄像头
- 操作系统: Windows/Linux

## 安装依赖


主要依赖包括：
- mediapipe
- opencv-python
- xgboost
- cupy-cuda11x (根据您的CUDA版本选择)
- ollama
- numpy
- tkinter

## 项目结构

```
emotionattack/
│
├── withgui.py              # 主程序 - 情感陪伴系统
├── onlinetest.py           # MediaPipe实时测试程序
├── test_ollama.py          # Ollama模型测试程序
├── Facial_Landmark_Extraction.py  # 面部特征提取程序
├── Data_Training.py        # 模型训练程序
│
└── model/
    └── mp_model.p          # 训练好的模型文件
```

## 使用说明

### 1. 启动情感陪伴系统

```bash
python withgui.py
```

### 2. 测试MediaPipe实时识别

```bash
python onlinetest.py
```

### 3. 测试Ollama模型

```bash
python test_ollama.py
```

### 4. 从数据集提取面部特征

```bash
python Facial_Landmark_Extraction.py
```

### 5. 训练模型

```bash
python Data_Training.py
```

## 训练过程

1. 准备数据集
2. 使用`Facial_Landmark_Extraction.py`提取面部特征
3. 运行`Data_Training.py`进行模型训练
4. 训练好的模型将保存在`model/mp_model.p`

## 模型性能

- 表情识别准确率: 86.5%
- 实时处理延迟: <100ms
- 支持识别的表情类别: 快乐、悲伤、惊讶

## 注意事项

1. 确保摄像头正常工作
2. 检查CUDA环境配置
3. 确保已正确安装Ollama并下载相应模型
4. 保持适当的光照条件以提高识别准确率

## 常见问题

1. **摄像头无法启动**
   - 检查摄像头连接
   - 确认摄像头权限设置

2. **模型加载失败**
   - 验证模型文件路径
   - 检查依赖包版本

3. **系统性能问题**
   - 确认GPU驱动更新
   - 检查CUDA版本兼容性

## 开发计划

- [ ] 扩展表情识别类别
- [ ] 优化模型性能
- [ ] 添加更多交互功能
- [ ] 开发移动端版本

## 贡献指南

欢迎提交问题和改进建议！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: xlxmatrix@qq.com
- GitHub Issues
