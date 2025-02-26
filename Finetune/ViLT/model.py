import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering, ViTModel, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import ViltConfig
from timm import create_model
from transformers import ViTModel, ViTFeatureExtractor
from models.image_encoder import load_image_encoder
import os
import math
from functools import partial
# from Mammo_clip.mammo_clip import Mammo_clip
from PIL import Image
import sys


celoss = nn.CrossEntropyLoss()
bceloss = nn.BCELoss()
class ViltWithBackbone(ViltForQuestionAnswering):
    def __init__(self, vision_backbone,config, data_info):
        super().__init__(config)
        self.backbone = vision_backbone
        self.classifiers = nn.ModuleDict({
            question_topic: nn.Linear(config.hidden_size, len(classes)) 
            for question_topic, classes in data_info.items()
        })

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        self.vilt.embeddings.patch_embeddings=self.backbone
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
        )
        exist_question_topic=list(labels.keys())
        logits = {}
        for question_topic, classifier in self.classifiers.items():
            if question_topic in exist_question_topic:
                if question_topic == "Abnormality":
                    logits[question_topic] = torch.sigmoid(classifier(outputs.last_hidden_state[:, 0, :]))
                else:
                    logits[question_topic] = classifier(outputs.last_hidden_state[:, 0, :])

        loss = 0.0
        for question_topic in exist_question_topic:
            if labels[question_topic] is not None:
                for question_topic, output in logits.items():
                    if question_topic in exist_question_topic:
                        if question_topic == "Abnormality":
                            loss += bceloss(output, labels[question_topic].unsqueeze(0))
                        else:
                            
                            loss += celoss(output, labels[question_topic].view(-1))
        return logits, loss
class Backbone(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone=backbone

    def forward(self, x):
        x=self.backbone(x)
        return x
class MultiTaskModel(nn.Module):
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""
    def __init__(self, data_info):
        super(MultiTaskModel, self).__init__()
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        backbone= Backbone(load_image_encoder('vitb14imagenet'))
        for param in backbone.parameters():
            param.requires_grad = False
        self.model = ViltWithBackbone(backbone,config=vilt_config, data_info=data_info)
    
    def forward(self, image,question,labels):
        encoding = self.processor(image, question, return_tensors="pt")
        pixel_values = torch.nn.functional.interpolate(encoding['pixel_values'],(224,224),mode='bilinear').to(self.model.device)
        
        input_ids = encoding['input_ids'].to(self.model.device)
        attention_mask = encoding['attention_mask'].to(self.model.device)

        for question_topic in labels:
            if labels[question_topic] is not None:
                labels[question_topic]=torch.tensor(labels[question_topic]).to(self.model.device)
        logits, loss = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,labels=labels)
        return logits, loss


