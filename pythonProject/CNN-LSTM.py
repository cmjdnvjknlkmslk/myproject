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

            X.append(padded_frames)
            y.append(gesture)

    X = np.array(X)
    y = np.array(y)
    y = label_encoder.transform(y)

    return X, y, label_encoder


# 设置参数
base_folder = "D:/cordinatesdata"   # 替换为实际的路径
max_frame_length = 80  # 假设最大帧数为60

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


# 定义 CNN-LSTM 模型
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.transpose(1, 2)  # 转换为 (batch, channels, sequence)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.transpose(1, 2)  # 转换回 (batch, sequence, features)

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

model = CNNLSTM(input_size, hidden_size, num_layers, num_classes, dropout_prob)
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
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


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


# # 示例预测
# example_csv = 'path_to_example_csv.csv'
# example_df = pd.read_csv(example_csv)
# example_frames = example_df[['Frame', 'X', 'Y']].values
# predicted_gesture = predict_gesture(model, example_frames)
# print(f"Predicted Gesture: {predicted_gesture}")
