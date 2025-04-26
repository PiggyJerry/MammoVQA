import torch
from PIL import Image
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'
sys.path.append(os.path.join(base_dir, 'Eval'))
from Utils import build_prompt
class MammoVQA_image_Bench(torch.utils.data.Dataset):
    def __init__(self, loaded_data, base_dir):
        self.loaded_data = loaded_data
        self.base_dir = base_dir
        # self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        sample = self.loaded_data[str(idx+1)]
        image_path = self.base_dir+sample['Path']
        ## question-answering-score
        qas_prompt= build_prompt(sample,score_type='question_answering_score')
        # cs_prompt = build_prompt(sample,score_type='certain_score')
        # image = Image.open(image_path).convert('RGB')
        # image = self.vis_processor(image)
        return image_path, qas_prompt, str(idx+1)
class MammoVQA_exam_Bench(torch.utils.data.Dataset):
    def __init__(self, loaded_data, base_dir):
        self.loaded_data = loaded_data
        self.base_dir = base_dir
        # self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        sample = self.loaded_data[str(idx+1)]
        image_path = [self.base_dir+path for path in sample['Path']]
        ## question-answering-score
        qas_prompt= build_prompt(sample,score_type='question_answering_score')
        # cs_prompt = build_prompt(sample,score_type='certain_score')
        # image = Image.open(image_path).convert('RGB')
        # image = self.vis_processor(image)
        return image_path, qas_prompt, str(idx+1)