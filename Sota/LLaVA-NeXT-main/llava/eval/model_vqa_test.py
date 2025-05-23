import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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

# 计算Eval目录的相对路径
# eval_dir = os.path.join(current_dir, '../../Eval')
# dataset_dir = os.path.join(current_dir, '../../Benchmark')
# 将Eval目录添加到sys.path
sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))
sys.path.append(os.path.join(base_dir, 'Sota/LLaVA-NeXT-main'))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

# 现在可以导入Certain_Score模块
from Certainty_Score import certainty_score
from Question_Answering_Score import question_answering_score
from Utils import build_prompt
from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench

# torch.cuda.set_device(3)
gpu_id='2'

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    
    # Model
    disable_torch_init()
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="for .*: copying from a non-meta parameter")
        # 加载模型的代码
        # model.load_state_dict(checkpoint)

        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device_map={"":'cuda:'+gpu_id})
        model=model.to('cuda:'+gpu_id)

        # idx = line["sample_id"]
        # question_type = line["metadata"]["question_type"]
        # dataset_name = line["metadata"]["dataset"]
        # gt = line["conversations"][1]["value"]
    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench-small.json'), 'r') as f:
        data = json.load(f)

    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)
    # yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    # Yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
    # _yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    # _Yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    results = {}
    for images, qas_questions, img_ids in tqdm(eval_dataloader):
        # image_file = images
        
        image_files = [images[0]]
        qas_qs="<image>"*len(image_files)+qas_questions[0]
    #     qs = "<image> "*len(image_files)+"This is a medical question about mammograph with yes/no options. Remember, answer only 'yes' or 'no' without any explanation. \
    # # The question is: Is the view of this mammograph is MLO?"
        qas_conversation=[
                {
                    "from": "human",
                    "value": qas_qs
                }
            ]
        # cur_prompt = args.extra_prompt + qs

        args.conv_mode = "qwen_1_5"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qas_qs)
        conv.append_message(conv.roles[1], None)

        qas_input_ids = preprocess_qwen([qas_conversation[0],{'from': 'gpt','value': None}], tokenizer, has_image=True).to('cuda:'+gpu_id)
        qas_img_num = list(qas_input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

        image_tensors = []
        for image_file in image_files:
            image = Image.open(image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().to('cuda:'+gpu_id))
        # image_tensors = torch.cat(image_tensors, dim=0)

        qas_stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        with torch.inference_mode():
            qas_output = model.generate(
                qas_input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                output_scores=True,return_dict_in_generate=True)
            
        qas_output_ids,qas_logits=qas_output.sequences,qas_output.scores

        qas_outputs = tokenizer.batch_decode(qas_output_ids, skip_special_tokens=True)[0]
        qas_outputs = qas_outputs.strip()
        if qas_outputs.endswith(qas_stop_str):
            qas_outputs = qas_outputs[:-len(qas_stop_str)]
        qas_outputs = qas_outputs.strip()
        print('text:',qas_outputs,len(qas_outputs))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/jiayi/MammoVQA/LLM/llava-next-interleave-qwen-7b")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)