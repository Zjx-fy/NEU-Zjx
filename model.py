import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, freeze_backbone=True):
    # 加载预训练的 ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        # 冻结所有卷积层
        for param in model.parameters():
            param.requires_grad = False
    
    # 替换最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model