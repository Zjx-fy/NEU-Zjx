import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 训练集数据增强（因为图片少，增强有助于泛化）
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证集仅做 resize 和归一化
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据集路径（根据你实际存放的位置）
data_dir = "data/dogs_vs_cats"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'val'),
    transform=val_transforms
)

# 因为图片数量很少，batch_size 设为 4（甚至 2）
batch_size = 4

# Windows 下 num_workers 必须为 0，否则报错
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 打印信息供检查
print(f"训练集图片数: {len(train_dataset)} (猫: {len(train_dataset)-len([i for i, l in enumerate(train_dataset.targets) if l==1])}, 狗: {len([i for i, l in enumerate(train_dataset.targets) if l==1])})")
print(f"验证集图片数: {len(val_dataset)}")
print(f"类别映射: {train_dataset.class_to_idx}")  # 应为 {'cat':0, 'dog':1}