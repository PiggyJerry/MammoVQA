import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

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

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default=os.path.join(base_dir, 'Sota/MiniGPT-4-main/eval_configs/minigpt4_eval.yaml'), help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# Model Initialization
conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
# setup_seeds()

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
# model.eval()
CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
    data = json.load(f)

MammoVQAData=MammoVQA_image_Bench(data,base_dir)
eval_dataloader = DataLoader(MammoVQAData, batch_size=16, shuffle=False)
results = {}
with open(base_dir + '/Result/minigpt-4.json', 'w') as f:
    for images, qas_questions, img_ids in tqdm(eval_dataloader):
        setup_seeds()
        chat_list = [CONV_VISION.copy() for _ in range(len(images))]
        qas_answers, qas_logits, qas_output_tokens = chat.batch_answer(images, qas_questions, chat_list, max_new_tokens=300)
        
        for qas_answer, qas_question, img_id in zip(qas_answers,  qas_questions,  img_ids):
            option, target, question_type = data[str(img_id)]['Options'], data[str(img_id)]['Answer'], data[str(img_id)]['Question type']
            qas_answer = qas_answer.replace('<unk>', '').replace('\u200b', '').replace('\n', '').strip()
        
            result = dict()
            result['qas_question'] = qas_question
            result['qas_answer'] = qas_answer

            results[str(img_id)] = result

        f.seek(0)
        f.truncate()
        json.dump(results, f, indent=4)
        f.write('\n')  # Add a newline to separate entries
        f.flush() 








