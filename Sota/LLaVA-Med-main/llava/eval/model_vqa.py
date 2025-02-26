import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math
from transformers import set_seed, logging

import sys
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'

sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))
sys.path.append(os.path.join(base_dir, 'Sota/LLaVA-Med-main'))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from Certainty_Score import certainty_score
from Question_Answering_Score import question_answering_score
from Utils import build_prompt
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench

logging.set_verbosity_error()
torch.cuda.set_device(6)
gpu_id='6'

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device_map={"":'cuda:'+gpu_id})
    model=model.to('cuda:'+gpu_id)
    for idx, param in enumerate(model.parameters()):
        print(f"Parameter {idx} is on device: {param.device}")


    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
        data = json.load(f)

    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)
    results = {}
    for images, qas_questions, img_ids in tqdm(eval_dataloader):
        # image_file = images
        qas_qs=qas_questions[0]
        if model.config.mm_use_im_start_end:
            qas_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qas_qs
        else:
            qas_qs = DEFAULT_IMAGE_TOKEN + '\n' + qas_qs

        qas_conv = conv_templates[args.conv_mode].copy()
        qas_conv.append_message(qas_conv.roles[0], qas_qs)
        qas_conv.append_message(qas_conv.roles[1], None)
        qas_prompt = qas_conv.get_prompt()

        qas_input_ids = tokenizer_image_token(qas_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda:'+gpu_id)

        image = Image.open(images[0])
        image_tensor = process_images([image], image_processor, model.config)[0]
        qas_stop_str = qas_conv.sep if qas_conv.sep_style != SeparatorStyle.TWO else qas_conv.sep2
        qas_keywords = [qas_stop_str]
        qas_stopping_criteria = KeywordsStoppingCriteria(qas_keywords, tokenizer, qas_input_ids)

        with torch.inference_mode():
            qas_output = model.generate(
                qas_input_ids,
                images=image_tensor.unsqueeze(0).half().to('cuda:'+gpu_id),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                output_scores=True,return_dict_in_generate=True)
            
         
            
        qas_output_ids,qas_logits=qas_output.sequences,qas_output.scores
        
        qas_answers = tokenizer.batch_decode(qas_output_ids, skip_special_tokens=True)
        
        for qas_answer, qas_question, img_id in zip(qas_answers, qas_questions, img_ids):

            target,question_type=data[str(img_id)]['Answer'],data[str(img_id)]['Question type']
            qas_answer=qas_answer.replace('<unk>','').replace('\u200b','').replace('\n','').strip()
            result = dict()
            result['qas_question']=qas_question
            result['qas_answer']=qas_answer
            results[str(img_id)] = result
    with open(base_dir+'/Result/LLAVA-Med.json', 'w') as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=os.path.join(base_dir, 'LLM/llava-med-v1.5-mistral-7b'))
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
