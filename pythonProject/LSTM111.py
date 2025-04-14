import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#数据增强
def augment_data(frames, num_augmentations=5):
    augmented_frames = []

    frames = frames.astype(np.float64)  # 确保数据是浮点数类型

    for _ in range(num_augmentations):
        # 添加噪声
        noisy_frames = frames + np.random.normal(0, 5, frames.shape)
        augmented_frames.append(noisy_frames)

        # 平移
        shift_x = np.random.randint(-5, 5)
        shift_y = np.random.randint(-5, 5)
        shifted_frames = frames.copy()
        shifted_frames[:, 1] += shift_x
        shifted_frames[:, 2] += shift_y
        augmented_frames.append(shifted_frames)

        # 缩放
        scale_factor = np.random.uniform(0.9, 1.1)
        scaled_frames = frames.copy()
        scaled_frames[:, 1:] *= scale_factor
        augmented_frames.append(scaled_frames)

        # 旋转
        angle = np.random.uniform(-10, 10) * np.pi / 180
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_frames = frames.copy()
        rotated_frames[:, 1:] = np.dot(frames[:, 1:], rotation_matrix)
        augmented_frames.append(rotated_frames)

        # 时间扰动
        jitter = np.random.uniform(-2, 2, frames[:, 0].shape)
        time_jittered_frames = frames.copy()
        time_jittered_frames[:, 0] += jitter
        augmented_frames.append(time_jittered_frames)

    return np.array(augmented_frames)


# 读取CSV文件并预处理数据
def load_data(base_folder, max_frame_length):
    X = []
    y = []
    label_encoder = LabelEncoder()

    # 文件夹和手势类型对应关系
    gesture_folders = {
        'clickpoint': 'click',
        'longpresspoint': 'long_press',
        'dragpoint': 'drag',
        'movepoint': 'move'
    }

    # 进行标签编码
    label_encoder.fit(list(gesture_folders.values()))


    for folder, gesture in gesture_folders.items():
        folder_path = os.path.join(base_folder, folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        print(f"Found {len(files)} files in {folder} for gesture {gesture}.")
        for file in files:
            df = pd.read_csv(file)
            frames = df[['Frame', 'X', 'Y']].values

            # 填充或截断
            if len(frames) < max_frame_length:
                padded_frames = np.pad(frames, ((0, max_frame_length - len(frames)), (0, 0)), 'constant')
            else:
                padded_frames = frames[:max_frame_length]

            # 原始数据
            X.append(padded_frames)
            y.append(gesture)

            # 数据增强
            augmented_frames = augment_data(padded_frames)
            for aug_frames in augmented_frames:
                X.append(aug_frames)
                y.append(gesture)

    X = np.array(X)
    y = np.array(y)
    y = label_encoder.transform(y)

    return X, y, label_encoder


# 设置参数
base_folder =  "D:/cordinatesdata" # 替换为实际的路径
max_frame_length = 60  # 假设最大帧数为60

# 加载数据
X, y, label_encoder = load_data(base_folder, max_frame_length)

# 检查数据是否为空
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check the base folder path and files.")

# 打印数据形状
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 确保 y 是 1D 数组
y = np.ravel(y)
print(f"y shape after ravel: {y.shape}")

# 划分训练集和测试集
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 检查划分后的数据集大小
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 检查Tensor的形状
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义改进的LSTM模型
class ImprovedGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob):
        super(ImprovedGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


input_size = 3
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y))
dropout_prob = 0.5

model = ImprovedGestureLSTM(input_size, hidden_size, num_layers, num_classes, dropout_prob)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


# 预测手势类型
def predict_gesture(model, frames):
    model.eval()
    if len(frames) < max_frame_length:
        padded_frames = np.pad(frames, ((0, max_frame_length - len(frames)), (0, 0)), 'constant')
    else:
        padded_frames = frames[:max_frame_length]

    padded_frames = torch.tensor(padded_frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(padded_frames)
        _, predicted = torch.max(outputs.data, 1)
        gesture_type = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return gesture_type


# 示例预测
example_csv = "D:\\cordinatesdata\\clickpoint\\0415.csv"
example_df = pd.read_csv(example_csv)
example_frames = example_df[['Frame', 'X', 'Y']].values
predicted_gesture = predict_gesture(model, example_frames)
print(f"Predicted Gesture: {predicted_gesture}")
