import json
import random

single_choice_prefix = "This is a mammography-related medical question with several options, only one of which is correct. \
Select the correct answer and respond with just the chosen option, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
multiple_choice_prefix = "This is a mammography-related medical question with several options, one or more of which may be correct. \
Select the correct answers and respond with only the chosen options, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
yesorno_prefix="This is a mammography-related medical question with 'Yes' or 'No' options. \
Respond with only 'Yes' or 'No' without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"

def build_prompt(sample,score_type):
    question_topic = sample['Question topic']
    question_type = sample['Question type']
    question = sample['Question']
    options = sample['Options']
    answer = sample['Answer']

    if score_type=='question_answering_score':
        if question_type == 'single choice':
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            prompt = single_choice_prefix.format(Question=question, Options=formatted_options)
        elif question_type == 'yes/no':
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            prompt = yesorno_prefix.format(Question=question, Options=formatted_options)
        else:
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            prompt = multiple_choice_prefix.format(Question=question, Options=formatted_options)
       
    return prompt
