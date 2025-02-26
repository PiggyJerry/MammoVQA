import json
import random
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('MammoVQA')[0]+'MammoVQA'

single_choice_prefix = "This is a mammography-related medical question with several options, only one of which is correct. \
Select the correct answer and respond with just the chosen option, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
multiple_choice_prefix = "This is a mammography-related medical question with several options, one or more of which may be correct. \
Select the correct answers and respond with only the chosen options, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
yesorno_prefix="This is a mammography-related medical question with 'Yes' or 'No' options. \
Respond with only 'Yes' or 'No' without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"

def build_prompt(sample):
    question_topic = sample['Question topic']
    question_type = sample['Question type']
    question = sample['Question']
    options = sample['Options']
    answer = sample['Answer']

    if question_type == 'single choice':
        hint = Hint.get(question_topic, "")
        random.shuffle(options)
        shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
        formatted_options = ", ".join(shuffled_options)
        # formatted_options=", ".join(options)
        prompt = single_choice_prefix.format(Question=question, Options=formatted_options, Hint=hint)
    else:
        hint = Hint.get(question_topic, "")
        random.shuffle(options)
        shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
        formatted_options = ", ".join(shuffled_options)
        # formatted_options=", ".join(options)
        prompt = multiple_choice_prefix.format(Question=question, Options=formatted_options, Hint=hint)
            
    return prompt
def process_answer(answer):
    if isinstance(answer, list):
        return ', '.join(answer)
    return answer

def generate_conversation(sample):
    image_files = [sample['Path']] if isinstance(sample['Path'], str) else sample['Path']
    prompt = build_prompt(sample)
    human_value = "<image>" * len(image_files) + prompt
    gpt_value = process_answer(sample['Answer'])
    
    return {
        "image": image_files,
        "conversations": [
            {
                "from": "human",
                "value": human_value
            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # 读取 JSON 文件
    file_paths = [
        f'{base_dir}/Benchmark/MammoVQA-Image-Eval.json',#f'{base_dir}/Benchmark/MammoVQA-Image-Train.json
        f'{base_dir}/Benchmark/MammoVQA-Exam-Eval.json'#f'{base_dir}/Benchmark/MammoVQA-Exam-Train.json'
    ]
    
    
    conversations = []
    
    for file_path in file_paths:
        data = load_json(file_path)
        
        for key, sample in data.items():
            conversation = generate_conversation(sample)
            conversations.append(conversation)
    
    # 保存生成的对话数据
    save_json(conversations, f'{base_dir}/lmms-finetune-main/Mammo-Eval.json')#f'{base_dir}/lmms-finetune-main/Mammo-Train.json'
    print(f"Generated JSON file saved to {base_dir}/lmms-finetune-main/Mammo-Eval.json")#{base_dir}/lmms-finetune-main/Mammo-Train.json"

if __name__ == "__main__":
    main()