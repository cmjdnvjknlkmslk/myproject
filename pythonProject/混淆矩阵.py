import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Defining the precision for each action
precision = {
    "Click": 0.98,
    "Double click": 0.96,
    "Slide": 0.96,
    "Long press": 0.98,
    "Drag": 0.98
}

# Creating a confusion matrix from precision values
# Assuming we have a relatively balanced dataset and the values correspond to true positive rates on diagonal
labels = list(precision.keys())
confusion_matrix = np.array([
    [precision["Click"], 0.01, 0.01, 0, 0],
    [0.02, precision["Double click"], 0, 0.02, 0],
    [0, 0, precision["Slide"], 0.02, 0.02],
    [0.01, 0.01, 0, precision["Long press"], 0],
    [0.01, 0, 0.01, 0, precision["Drag"]]
])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt=".2f")
plt.title("Confusion Matrix for Action Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
