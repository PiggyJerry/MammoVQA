import os
import time
import numpy as np
from skimage import io
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision
# from Mammo_VQA_dataset import MammoVQA_image,custom_collate_fn
from torch.utils.data import DataLoader
# from basics import f1_mae_torch, dice_torch #normPRED, GOSPRF1ScoresCache,f1score_torch,
from torchvision import transforms
from model import MultiTaskModel

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import re
import contextlib
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import json
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'
import sys
sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))
sys.path.append(os.path.join(base_dir, 'Sota/LLaVA-NeXT-main'))

from Certainty_Score import certainty_score
from Question_Answering_Score import question_answering_score
from Utils import build_prompt
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench


def valid(net, hypar, epoch=0):
    net.eval()
    # print("Validating...")

    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
        data = json.load(f)

    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)

    results = {}
    for images, qas_questions, img_ids in tqdm(eval_dataloader):
        # image_file = images
        
        image_files = [images[0]]
        image_list = []
        for image_file in image_files:
            image = Image.open(image_file).convert('RGB')
            image_list.append(image)
        question=[qas_questions[0].split('### Question: ')[-1].split(' ### Options')[0]]
        question_topic=data[img_ids[0]]['Question topic']
        label={question_topic:None}
        logits, loss = net(image_list, question,label)
        logit=logits[question_topic]
        if question_topic!='Abnormality':
            single=torch.argmax(logit).item()
            output=[hypar['reverse_label_mappings'][question_topic][single]]
        else:
            multiple=(logit> 0.5).float()
            indices = torch.where(multiple == 1)[1]
            multiple=indices.tolist()
            choices=[]
            for index in multiple:
                choices.append(hypar['reverse_label_mappings'][question_topic][index])
            output=[', '.join(choices)]
        
        
        
        for qas_answer, qas_question, img_id in zip(output, question, img_ids):
            result = dict()
            result['qas_question']=qas_question
            result['qas_answer']=qas_answer
            results[str(img_id)] = result
        
    with open(base_dir+'/Result/ViLT_withbackbone.json', 'w') as f:
        json.dump(results, f, indent=4)

def main(hypar): 
    net = hypar["model"]
    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        if len(hypar['gpu_id']) > 1:
            net = net.cuda(hypar['gpu_id'][0])
            net = nn.DataParallel(net, device_ids=hypar['gpu_id'])
        else:
            net = net.cuda(hypar['gpu_id'][0])
            
    if(hypar["restore_model"]!=""):
        print("restore model from:")
        print(hypar["restore_model"])
        if torch.cuda.is_available():
            if len(hypar['gpu_id']) > 1:
                net.load_state_dict(torch.load(hypar["restore_model"], map_location=lambda storage, loc: storage.cuda(hypar['gpu_id'][0])))
            else:
                state_dict = net.state_dict()
                for param_name in state_dict.keys():
                    print(param_name)
                # model = model.cuda(hypar['gpu_id'][0])
                pretrained_dict = torch.load(hypar["restore_model"], map_location=lambda storage, loc: storage.cuda(hypar['gpu_id'][0]))
                # pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
                net.load_state_dict(pretrained_dict, strict=False)
        else:
            net.load_state_dict(torch.load(hypar["restore_model"], map_location='cpu'))
    valid(net, hypar)


if __name__ == "__main__":

    hypar = {}
    hypar["mode"] = "eval"
    hypar['finetune']='lp'
    hypar['gpu_id']=[1]
    
    ## -- 2.2. choose floating point accuracy --
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    ## --- 2.4. data augmentation parameters ---
    hypar["input_size"] = [224, 224] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["model_path"]=f"{base_dir}/Finetune/ViLT/saved_model/"

    ## --- 2.5. define model  ---
    # print("building model...")
    data_info = {
        'View': {'MLO':0,'CC':1},
        'Laterality': {'Right':0,'Left':1},
        'Pathology': {'Normal':0,'Malignant':1,'Benign':2},
        'Background tissue': {'Fatty-glandular':0,'Fatty':1,'Dense-glandular':2},
        'ACR': {'Level A':0,'Level B':1,'Level C':2,'Level D':3},
        'Subtlety': {'Normal':0,'Level 1':1,'Level 2':2,'Level 3':3,'Level 4':4,'Level 5':5},
        'Bi-Rads': {'Bi-Rads 0':0,'Bi-Rads 1':1,'Bi-Rads 2':2,'Bi-Rads 3':3,'Bi-Rads 4':4,'Bi-Rads 5':5,'Bi-Rads 6':6},
        'Masking potential': {'Level 1':0,'Level 2':1,'Level 3':2,'Level 4':3,'Level 5':4,'Level 6':5,'Level 7':6,'Level 8':7},
        'Abnormality': {'Architectural':0,'Asymmetry':1,'Calcification':2,'Mass':3,'Miscellaneous':4,'Nipple retraction':5,'Normal':6,'Skin retraction':7,'Skin thickening':8,'Suspicious lymph node':9},
    }

    hypar['label_mappings'] = data_info
    def create_reverse_label_mapping(data_info):
        """
        Creates a reverse mapping from index to label for each category in data_info.

        Args:
            data_info (dict): A dictionary containing label mappings for various categories.

        Returns:
            dict: A dictionary containing reverse mappings for each category.
        """
        reverse_label_mapping = {}

        for category, mapping in data_info.items():
            # Reverse the label mapping for the current category
            reverse_label_mapping[category] = {v: k for k, v in mapping.items()}

        return reverse_label_mapping

    hypar['reverse_label_mappings'] = create_reverse_label_mapping(data_info)
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""

    hypar["batch_size_train"] = 1 ## batch size for training
    hypar["grad_accumulate"]=16
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    hypar["restore_model"]=hypar['model_path']+"ViLT_Mammo.pth"
    hypar["model"]=MultiTaskModel(hypar['label_mappings'])
    main(hypar=hypar)