import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   
import torch.nn as nn

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
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench

def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(qas_question,image_list,image_padding_tokens):
    
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    qas_new_qestions = [_ for _ in qas_question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        qas_new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + qas_new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    qas_text = ''.join(qas_new_qestions) 
    return qas_text, vision_x, 
    
    
def main():
    print("Setup tokenizer")
    text_tokenizer,image_padding_tokens = get_tokenizer(os.path.join(current_dir, 'Language_files'))
    print("Finish loading tokenizer")
    
    ### Initialize a simple case for demo ###
    
    print("Setup Model")
    from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map

    model = MultiLLaMAForCausalLM(
        lang_model_path=os.path.join(current_dir, 'Language_files'),
    )
    print("Finish loading model")
    
    # num_gpus=2
    
    max_memory = {3: "40GiB", 4: "40GiB"}
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["MultiLLaMAForCausalLM"])


    model = load_checkpoint_and_dispatch(
        model, checkpoint=os.path.join(current_dir, 'pytorch_model.bin'), device_map=device_map,
    )

    model.eval()
    
    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
        data = json.load(f)

    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)
    results = {}
    with torch.no_grad():
        for images, qas_questions, img_ids in tqdm(eval_dataloader):
            image =[
                    {
                        'img_path': images[0],
                        'position': 0, #indicate where to put the images in the text string, range from [0,len(question)-1]
                    }, # can add abitrary number of imgs
                ] 
                
            qas_text,vision_x = combine_and_preprocess(qas_questions[0],image,image_padding_tokens)    
            qas_lang_x = text_tokenizer(
                qas_text, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids']
          
            qas_lang_x = qas_lang_x.to('cuda')
            vision_x = vision_x.to('cuda')

            qas_generation = model.generate(qas_lang_x, vision_x)
            qas_batch_output_tokens=qas_generation.sequences
            qas_answers = text_tokenizer.batch_decode(qas_batch_output_tokens, skip_special_tokens=True)
           
            
            for qas_answer, qas_question, img_id in zip(qas_answers, qas_questions, img_ids):
                qas_answer=qas_answer.replace('<unk>','').replace('\u200b','').replace('\n','').strip()
               
                result = dict()
                result['qas_question']=qas_question
                result['qas_answer']=qas_answer
                results[str(img_id)] = result
        with open(base_dir+'/Result/RadFM.json', 'w') as f:
            json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    main()
       
