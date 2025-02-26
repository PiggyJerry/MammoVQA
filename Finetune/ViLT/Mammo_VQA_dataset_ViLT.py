import torch
from PIL import Image
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'
class MammoVQA_image(torch.utils.data.Dataset):
    def __init__(self, loaded_data,label_mappings):
        self.loaded_data = loaded_data
        self.label_mappings=label_mappings
        self.type=type

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        sample = self.loaded_data[idx]
        image_path = base_dir+sample['Path']
        image=Image.open(image_path).convert('RGB')
        question=sample['Question']
        label=sample['Answer']
        question_topic=sample['Question topic']
        if question_topic!="Abnormality":
            label = {question_topic:self.label_mappings[question_topic][sample['Answer']]}
        else:
            label_mapping = self.label_mappings[question_topic]
            label = {question_topic:np.zeros(len(label_mapping), dtype=np.float32)}  # 初始化一个全零的 label 向量
            
            for finding in sample['Answer']:
                if finding in label_mapping:  # 如果答案在映射中，设置相应位置为 1.0
                    label[question_topic][label_mapping[finding]] = 1.0

        ids=sample['ID']
        return image, question, label, ids
def custom_collate_fn(batch):
    images, questions, labels, ids = zip(*batch)
    return list(images), list(questions), list(labels), list(ids)