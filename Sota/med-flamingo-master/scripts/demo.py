from huggingface_hub import hf_hub_download
import torch
import os

from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys

from demo_utils import image_paths, clean_generation

import sys
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'
sys.path.append(os.path.join(base_dir, 'Eval'))
sys.path.append(os.path.join(base_dir, 'Benchmark'))
sys.path.append(os.path.join(base_dir, 'Sota/med-flamingo-master'))
sys.path.append(os.path.join(base_dir, 'Sota/med-flamingo-master/open_flamingo-main/open_flamingo/src'))
from factory import create_model_and_transforms
from src.utils import FlamingoProcessor

from PIL import Image
from Mammo_VQA_dataset import MammoVQA_image_Bench

torch.cuda.set_device(7)

def main():
    accelerator = Accelerator() #when using cpu: cpu=True

    device = accelerator.device
    
    print('Loading model..')

    # >>> add your local path to Llama-7B (v1) model here:
    llama_path = os.path.join(base_dir, 'LLM/decapoda-research-llama-7B-hf')
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()
    
    with open(os.path.join(base_dir, 'Benchmark/MammoVQA-Image-Bench.json'), 'r') as f:
        data = json.load(f)

    MammoVQAData=MammoVQA_image_Bench(data,base_dir)
    eval_dataloader = DataLoader(MammoVQAData, batch_size=1, shuffle=False)
    results = {}
    for images, qas_questions, img_ids in tqdm(eval_dataloader):
        

        """
        Step 1: Load images
        """
        demo_images = [Image.open(images[0])]
        qas_question=qas_questions[0].replace("### Question", "<image>"*len(demo_images)+"### Question")

        """
        Step 2: Define multimodal few-shot prompt 
        """

        """
        Step 3: Preprocess data 
        """
        print('Preprocess data')
        pixels = processor.preprocess_images(demo_images)
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
        qas_tokenized_data = processor.encode_text(qas_question)
        """
        Step 4: Generate response 
        """

        # actually run few-shot prompt through model:
        print('Generate from multimodal few-shot prompt')
        qas_generate = model.generate(
            vision_x=pixels.to(device),
            lang_x=qas_tokenized_data["input_ids"].to(device),
            attention_mask=qas_tokenized_data["attention_mask"].to(device),
            max_new_tokens=20,
        )
      
        qas_generated_text,qas_logits=qas_generate.sequences,qas_generate.scores
        qas_response = processor.tokenizer.decode(qas_generated_text[0])
        qas_response = clean_generation(qas_response)
        qas_answers = [qas_response.split('### Answer:')[1]]
       
        for qas_answer, qas_question,img_id in zip(qas_answers, qas_questions, img_ids):
            target,question_type=data[str(img_id)]['Answer'],data[str(img_id)]['Question type']
            qas_answer=qas_answer.replace('<unk>','').replace('\u200b','').replace('\n','').strip()
           
            result = dict()
            result['qas_question']=qas_question
            result['qas_answer']=qas_answer
            results[str(img_id)] = result
    with open(base_dir+'/Result/Med-Flamingo.json', 'w') as f:
        json.dump(results, f, indent=4)
        

if __name__ == "__main__":

    main()
