import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.image_encoder import load_image_encoder
celoss = nn.CrossEntropyLoss()
bceloss = nn.BCELoss()
class Backbone(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone=backbone

    def forward(self, x):
        x=self.backbone(x)
        return x
class DINOv2MultiTask(nn.Module):
    def __init__(self, data_info):
        super().__init__()
        
        self.dinov2 = Backbone(load_image_encoder('vitb14imagenet'))
        for param in self.dinov2.parameters():
            param.requires_grad = False

        self.feature_dim = 768

        self.classifiers = nn.ModuleDict({
            question_topic: nn.Linear(self.feature_dim, len(classes)) 
            for question_topic, classes in data_info.items()
        })

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images, labels=None):
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images[0], Image.Image):
            processed_images = torch.stack([self.transform(img) for img in images])
        else:
            processed_images = images

        device = next(self.parameters()).device
        processed_images = processed_images.to(device)

        features = self.dinov2(processed_images)

        logits = {}
        for question_topic, classifier in self.classifiers.items():
            if question_topic == "Abnormality":
                logits[question_topic] = torch.sigmoid(classifier(features))
            else:
                logits[question_topic] = classifier(features)

        loss = 0.0
        if labels is not None:
            if isinstance(labels, dict):
                labels = [labels]

            processed_labels = {}
            for batch_label in labels:
                if isinstance(batch_label, dict):
                    for question_topic, label_value in batch_label.items():
                        if question_topic not in processed_labels:
                            processed_labels[question_topic] = []

                        if isinstance(label_value, (int, float)):
                            label_value = torch.tensor(label_value)
                        elif isinstance(label_value, np.ndarray):
                            label_value = torch.from_numpy(label_value)
                        
                        processed_labels[question_topic].append(label_value)
            
            for question_topic, label_values in processed_labels.items():
                if label_values and label_values[0] is not None:
                    label_tensor = torch.stack(label_values).to(device)
                    if question_topic == "Abnormality":
                        loss += bceloss(logits[question_topic], label_tensor.float())
                    else:
                        loss += celoss(logits[question_topic], label_tensor.view(-1))
        
        return logits, loss

class MultiTaskModel(nn.Module):
    def __init__(self, data_info):
        super(MultiTaskModel, self).__init__()
        self.model = DINOv2MultiTask(data_info)
    
    def forward(self, images, labels=None):
        logits, loss = self.model(images=images, labels=labels)
        return logits, loss