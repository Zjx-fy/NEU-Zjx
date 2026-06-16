import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import get_model
from data_loader import val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=2, freeze_backbone=True)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

target_names = ['cat', 'dog']

# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("混淆矩阵:")
print(cm)

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 显示部分验证集图像及其预测结果（可选）
from data_loader import val_transforms   # 用于反归一化显示
def imshow(img_tensor, title):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

dataiter = iter(val_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12, 8))
for idx in range(min(8, len(images))):
    plt.subplot(2, 4, idx+1)
    imshow(images[idx].cpu(), f"True: {target_names[labels[idx]]}\nPred: {target_names[preds[idx]]}")
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300)
plt.show()