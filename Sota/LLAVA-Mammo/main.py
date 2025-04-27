from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import argparse
import torch
import os
import json
from tqdm import tqdm

from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

import sys
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'

sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))

from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench

# torch.cuda.set_device(3)
gpu_id='1'

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(f"{base_dir}/Finetune/lmms-finetune-main/checkpoints/llava-1.6-vicuna-7b_lora-True_qlora-False", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:"+gpu_id)

with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
    data = json.load(f)

MammoVQAData=MammoVQA_image_Bench(data,base_dir)
eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)

results = {}
for images, qas_questions, img_ids in tqdm(eval_dataloader):
    
    image_files = [images[0]]
    image_list = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image_list.append(image)
    qas_qs=qas_questions[0]
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": qas_qs}] + \
                        [{"type": "image"}] * len(image_files),
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image_list, prompt, return_tensors="pt").to("cuda:"+gpu_id)

    # autoregressively complete prompt
    output_ids = model.generate(**inputs, max_new_tokens=100)

    qas_outputs=[processor.decode(output_ids[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()]
    
    for qas_answer, qas_question, img_id in zip(qas_outputs, qas_questions, img_ids):
        qas_answer=qas_answer.replace('<unk>','').replace('\u200b','').replace('\n','').strip()
        result = dict()
        result['qas_question']=qas_question
        result['qas_answer']=qas_answer
        results[str(img_id)] = result
    
with open(base_dir+'/Result/LLaVA-Mammo.json', 'w') as f:
    json.dump(results, f, indent=4)