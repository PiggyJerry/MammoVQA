import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultiTaskModel
from PIL import Image
import json
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'
import sys
sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))

from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench


def valid(net, hypar, epoch=0):
    net.eval()

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
        logits, loss = net(image,label)
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
        
    with open(base_dir+'/Result/DiNOv2.json', 'w') as f:
        json.dump(results, f, indent=4)

def main(hypar):
    net = hypar["model"]

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
                pretrained_dict = torch.load(hypar["restore_model"], map_location=lambda storage, loc: storage.cuda(hypar['gpu_id'][0]))
                net.load_state_dict(pretrained_dict, strict=False)
        else:
            net.load_state_dict(torch.load(hypar["restore_model"], map_location='cpu'))
   
    valid(net, hypar)


if __name__ == "__main__":
    hypar = {}
    hypar["mode"] = "eval"
    hypar['finetune']='lp'#lp or ft
    hypar['gpu_id']=[0]
    
    hypar["model_digit"] = "full" 
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar["input_size"] = [224, 224] 
        
    hypar["model_path"]=f"{current_dir}/saved_model/"

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
  
    hypar["batch_size_train"] = 1 ## batch size for training
    hypar["grad_accumulate"]=16
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    hypar["restore_model"]=hypar['model_path']+"DiNOv2.pth"
    hypar["model"]=MultiTaskModel(hypar['label_mappings'])
    main(hypar=hypar)