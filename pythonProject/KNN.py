import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 数据增强函数
def augment_data(frames, num_augmentations=5):
    augmented_frames = []
    frames = frames.astype(np.float64)

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

# 数据加载和增强函数
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

# 设置参数
base_folder = "D:/cordinatesdata"    # 替换为实际的路径
max_frame_length = 80  # 假设最大帧数为80

# 加载和增强数据
X, y, label_encoder = load_and_augment_data(base_folder, max_frame_length)

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

# 平坦化输入特征
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 定义KNN模型
knn = KNeighborsClassifier(n_neighbors=5)

# 训练KNN模型
knn.fit(X_train_flat, y_train)

# 预测并评估模型
y_train_pred = knn.predict(X_train_flat)
y_test_pred = knn.predict(X_test_flat)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# 计算分类报告
train_report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

# 绘制分类报告
def plot_classification_report(report, title='Classification Report', cmap='viridis'):
    classes = list(report.keys())
    classes.remove('accuracy')
    classes.remove('macro avg')
    classes.remove('weighted avg')

    plot_mat = []
    for cls in classes:
        metrics = report[cls]
        plot_mat.append([metrics['precision'], metrics['recall'], metrics['f1-score']])

    fig, ax = plt.subplots()
    cax = ax.matshow(plot_mat, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.show()

plot_classification_report(train_report, title='Training Classification Report')
plot_classification_report(test_report, title='Testing Classification Report')

# 打印每种动作的识别准确率
def print_class_accuracy(report, dataset_type=''):
    print(f"\n{dataset_type} Classification Accuracy per Class:")
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            accuracy = metrics['recall']
            print(f"{cls}: {accuracy * 100:.2f}%")

print_class_accuracy(train_report, 'Training')
print_class_accuracy(test_report, 'Testing')

# 计算距离示例
def calculate_distances(X_train, X_test_sample):
    distances = []
    for train_sample in X_train:
        dist = np.linalg.norm(train_sample - X_test_sample)
        distances.append(dist)
    return distances

# 示例计算一个测试样本与所有训练样本的距离
sample_test_index = 0
distances = calculate_distances(X_train_flat, X_test_flat[sample_test_index])
print(f"Distances from test sample {sample_test_index} to all training samples: {distances}")