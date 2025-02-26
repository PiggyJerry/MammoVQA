from PIL import Image

import torch
from transformers import LlamaTokenizer

from src.model.internvl_chat import InternVLChatModel
from src.dataset.transforms import build_transform
import sys
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'

sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))
sys.path.append(os.path.join(base_dir, 'Sota/LLaVA-NeXT-main'))

# 现在可以导入Certain_Score模块
from Certainty_Score import certainty_score
from Question_Answering_Score import question_answering_score
from Utils import build_prompt
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

model_path = f"{base_dir}/LLM/MedDr"

device = "cuda"

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)
from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map
model = InternVLChatModel.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
max_memory = {3: "40GiB", 4: "40GiB"}
device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["InternVLChatModel"])

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = load_checkpoint_and_dispatch(
    model=model,  
    checkpoint=model_path,  
    device_map=device_map, 
).eval()

image_size = model.config.force_image_size or model.config.vision_config.image_size
pad2square = model.config.pad2square
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
    data = json.load(f)

MammoVQAData=MammoVQA_image_Bench(data,base_dir)
eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)

results = {}
for images, qas_questions, img_ids in tqdm(eval_dataloader):
    # image_file = images
    
    image = Image.open(images[0]).convert('RGB')
    image = image_processor(image).unsqueeze(0).to(device).to(torch.bfloat16)
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=image,
            question=qas_questions[0],
            generation_config=generation_config,
            print_out=False,
            
        )
    
    for qas_answer, qas_question, img_id in zip(response, qas_questions, img_ids):

        qas_answer=qas_answer.replace('<unk>','').replace('\u200b','').replace('\n','').strip()
      
        result = dict()
        result['qas_question']=qas_question
        result['qas_answer']=qas_answer
        results[str(img_id)] = result
    
with open(base_dir+'/Result/MedDr.json', 'w') as f:
    json.dump(results, f, indent=4)


