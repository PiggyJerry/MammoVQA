import argparse
import os
import csv
import json
import math
import numpy as np
from tqdm import tqdm
from typing import Optional
import difflib 
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from torch import nn
from torch.utils.data import DataLoader  
import torch
# from tensorboardX import SummaryWriter
from torch.nn import functional as F
from transformers import LlamaTokenizer
from models.QA_model import QA_model
import sys
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'

sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))

from Certainty_Score import certainty_score
from Question_Answering_Score import question_answering_score
from Utils import build_prompt
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench
torch.cuda.set_device(2)
gpu_id='2'
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default=os.path.join(base_dir,"Sota/MedVInT_TD/Results/PMC-LLAMA"))
    ckp: Optional[str] = field(default=os.path.join(base_dir,"Sota/MedVInT_TD/Results"))
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default=os.path.join(base_dir,"Sota/MedVInT_TD/Results/checkpoint.pt"))
    
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    # img_dir: str = field(default='', metadata={"help": "Path to the training data."})
    # Test_csv_path: str = field(default='./Data/final_train/final_test.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default=os.path.join(base_dir,"Sota/MedVInT_TD/Results/PMC-LLAMA/tokenizer.model"), metadata={"help": "Path to the training data."})
    trier: int = field(default=0)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default=os.path.join(base_dir,"Sota/MedVInT_TD/Results"))
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
  
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    
    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    ckpt = torch.load(ckp, map_location='cpu')
    for name in list(ckpt.keys()):
        if 'self_attn.q_proj.weight' in name and "vision_model" not in name:
            new_name = name.replace('self_attn.q_proj.weight', 'self_attn.q_proj.base_layer.weight')
            ckpt[new_name] = ckpt.pop(name)
        if 'self_attn.v_proj.weight' in name and "vision_model" not in name:
            new_name = name.replace('self_attn.v_proj.weight', 'self_attn.v_proj.base_layer.weight')
            ckpt[new_name] = ckpt.pop(name)
        if 'lora_A' in name:
            new_name = name.replace('lora_A', 'lora_A.default')
            ckpt[new_name] = ckpt.pop(name)
        if 'lora_B' in name:
            new_name = name.replace('lora_B', 'lora_B.default')
            ckpt[new_name] = ckpt.pop(name)

    model = QA_model(model_args)

    model.load_state_dict(ckpt)
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(base_dir,"Sota/MedVInT_TD/Results/PMC-LLAMA/tokenizer.model"))
    tokenizer.pad_token_id=0
    tokenizer.eos_token_id=1
    
    print("Start Testing")
    
    model = model.to('cuda:'+gpu_id)
    model.eval()
    
    from torchvision import transforms
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512]),   
                transforms.ToTensor(),
                normalize,
            ])
    
    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
        data = json.load(f)
    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)
  
    results = {}
    with torch.no_grad():
        for images, qas_questions, img_ids in tqdm(eval_dataloader):
            image = Image.open(images[0]).convert('RGB')   
            image = transform(image)
            image = image.unsqueeze(0).to('cuda:'+gpu_id)

            qas_input_ids = tokenizer(qas_questions[0],return_tensors="pt").to('cuda:'+gpu_id)
            
            with torch.no_grad():
                qas_generation = model.generate_long_sentence(qas_input_ids['input_ids'],image)
                
            qas_generation_ids,qas_logits=qas_generation.sequences,qas_generation.scores
            qas_answers = tokenizer.batch_decode(qas_generation_ids, skip_special_tokens=True)
          
            for qas_answer, qas_question,img_id in zip(qas_answers, qas_questions, img_ids):
                target,question_type=data[str(img_id)]['Answer'],data[str(img_id)]['Question type']
              
                result = dict()
                result['qas_question']=qas_question
                result['qas_answer']=qas_answer
                results[str(img_id)] = result
        with open(base_dir+'/Result/MedVInT_TD.json', 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

