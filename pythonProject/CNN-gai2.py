import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def augment_data(frames, num_augmentations=5):
    augmented_frames = []

    frames = frames.astype(np.float64)

    for _ in range(num_augmentations):
        noisy_frames = frames + np.random.normal(0, 5, frames.shape)
        augmented_frames.append(noisy_frames)

        shift_x = np.random.randint(-5, 5)
        shift_y = np.random.randint(-5, 5)
        shifted_frames = frames.copy()
        shifted_frames[:, 1] += shift_x
        shifted_frames[:, 2] += shift_y
        augmented_frames.append(shifted_frames)

        scale_factor = np.random.uniform(0.9, 1.1)
        scaled_frames = frames.copy()
        scaled_frames[:, 1:] *= scale_factor
        augmented_frames.append(scaled_frames)

        angle = np.random.uniform(-10, 10) * np.pi / 180
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_frames = frames.copy()
        rotated_frames[:, 1:] = np.dot(frames[:, 1:], rotation_matrix)
        augmented_frames.append(rotated_frames)

        jitter = np.random.uniform(-2, 2, frames[:, 0].shape)
        time_jittered_frames = frames.copy()
        time_jittered_frames[:, 0] += jitter
        augmented_frames.append(time_jittered_frames)

    return np.array(augmented_frames)

def load_and_augment_data(base_folder, max_frame_length):
    X = []
    y = []
    label_encoder = LabelEncoder()

    gesture_folders = {
        'clickpoint': 'click',
        'longpresspoint': 'long_press',
        'dragpoint': 'drag',
        'movepoint': 'move'
    }

    label_encoder.fit(list(gesture_folders.values()))

    for folder, gesture in gesture_folders.items():
        folder_path = os.path.join(base_folder, folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        print(f"Found {len(files)} files in {folder} for gesture {gesture}.")
        for file in files:
            df = pd.read_csv(file)
            frames = df[['Frame', 'X', 'Y']].values

            if len(frames) < max_frame_length:
                padded_frames = np.pad(frames, ((0, max_frame_length - len(frames)), (0, 0)), 'constant')
            else:
                padded_frames = frames[:max_frame_length]

            X.append(padded_frames)
            y.append(gesture)

            augmented_frames = augment_data(padded_frames)
            for aug_frames in augmented_frames:
                X.append(aug_frames)
                y.append(gesture)

    X = np.array(X)
    y = np.array(y)
    y = label_encoder.transform(y)

    return X, y, label_encoder

def standardize_data(X):
    scaler = StandardScaler()
    num_samples, num_frames, num_features = X.shape
    X = X.reshape(-1, num_features)
    X = scaler.fit_transform(X)
    X = X.reshape(num_samples, num_frames, num_features)
    return X

base_folder = "D:/cordinatesdata"
max_frame_length = 80

X, y, label_encoder = load_and_augment_data(base_folder, max_frame_length)
X = standardize_data(X)

if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check the base folder path and files.")

print(f"X shape: {X.shape}, y shape: {y.shape}")

y = np.ravel(y)
print(f"y shape after ravel: {y.shape}")

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# class SimpleCNN(nn.Module):
#     def __init__(self, input_size, num_classes, dropout_prob):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(32 * (max_frame_length // 2 // 2), 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_prob)
class CNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (max_frame_length // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_size = 3
num_classes = len(np.unique(y))
dropout_prob = 0.5

#model = SimpleCNN(input_size, num_classes, dropout_prob)
model = CNN(input_size, num_classes, dropout_prob)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

num_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')
early_stop_count = 0
early_stop_patience = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {val_accuracy:.2f}%")

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     early_stop_count = 0
    # else:
    #     early_stop_count += 1
    #     if early_stop_count >= early_stop_patience:
    #         print("Early stopping")
    #         break

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

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()

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

example_csv = "D:\\cordinatesdata\\clickpoint\\0415.csv"
example_df = pd.read_csv(example_csv)
example_frames = example_df[['Frame', 'X', 'Y']].values
predicted_gesture = predict_gesture(model, example_frames)
print(f"Predicted Gesture: {predicted_gesture}")
