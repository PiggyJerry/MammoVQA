import json
import random

# 定义提示模板和提示信息
Hint = {
    "View": "A mammogram's view type refers to the specific angle or position from which a breast X-ray is taken. - MLO (Mediolateral Oblique): This view is taken from an angle. - CC (Cranio-Caudal): This view is taken from above the breast, with the X-ray moving vertically from the top (cranial) to the bottom (caudal) of the breast.",
    "Laterality": "Laterality refers to the side of the body on which a mammogram is performed. - Right: The mammogram image taken from the right side of the body. - Left: The mammogram image taken from the left side of the body.",
    "ACR": "ACR stands for the American College of Radiology, which provides a classification system for mammogram findings. - Level A: The breast is almost entirely fatty. - Level B: There are scattered areas of fibroglandular density. - Level C: The breasts are heterogeneously dense, which may obscure small masses. - Level D: The breasts are extremely dense, which lowers the sensitivity of mammography.",
    "Bi-Rads": "Bi-Rads is a system used to categorize the findings on mammograms into levels that describe the risk of cancer. - Bi-Rads 0: Need additional imaging evaluation. - Bi-Rads 1: Negative. - Bi-Rads 2: Benign finding. - Bi-Rads 3: Probably benign finding. - Bi-Rads 4: Suspicious abnormality. - Bi-Rads 5: Highly suggestive of malignancy. - Bi-Rads 6: Known biopsy-proven malignancy.",
    "Background tissue": "Background tissue describes the type of breast tissue visible on mammograms. - Fatty-glandular: Indicates a mix of fatty and fibrous tissues. - Fatty: Majority of the breast tissue is fat, which may make it easier to spot abnormalities. - Dense-glandular: Indicates high density of glandular tissue, which can sometimes make it hard to detect tumors due to less fat.",
    "Abnormality": "Abnormalities in mammograms are any unusual or atypical finding in the breast tissue that differs from the normal appearance. - Normal: No abnormalities. - Calcification: Small calcium deposits. - Mass: A space-occupying lesion. - Architectural distortion: Distortion of normal breast structure. - Asymmetry: Unequal tissue density. - Miscellaneous: Other rare findings. - Nipple retraction: Inward-turning nipple. - Suspicious lymph node: Unusual lymph nodes. - Skin thickening: Thickened breast skin. - Skin retraction: Skin pulling inward.",
    "Pathology": "Pathology of a mammogram indicates the nature of a detected abnormality. - Normal: No malignant findings. - Malignant: Cancerous growths that require further medical action. - Benign: Non-cancerous growths that are typically not harmful.",
    "Subtlety": "Subtlety refers to how obvious the abnormal findings are on a mammogram. - Normal: No abnormalities. - Level 1: Extremely subtle. - Level 2: Moderately subtle. - Level 3: Mildly subtle. - Level 4: Fairly obvious. - Level 5: Obvious.",
    "Cancer": "Cancer refers to the presence or absence of malignant tumor in the breast.",
    "Invasive cancer": "Invasive cancer indicates whether the cancer has spread from the original tissue to other parts of the body.",
    "Masking potential": "Masking potential indicates the possibility for cancer to be obscured in a mammogram. - Level 1 to Level 8: Lowest potential to be obscured to highest potential to be obscured."
}


# single_choice_prefix = "This is a mammography-related medical question with several options, only one of which is correct. \
# Select the correct answer and respond with just the chosen option, without any further explanation. \
# ### Question: {Question} ### Options: {Options}. ### Answer:"
# multiple_choice_prefix = "This is a mammography-related medical question with several options, one or more of which may be correct. \
# Select the correct answers and respond with only the chosen options, without any further explanation. \
# ### Question: {Question} ### Options: {Options}. ### Answer:"
# yesorno_prefix="This is a mammography-related medical question with 'Yes' or 'No' options. \
# Respond with only 'Yes' or 'No' without any further explanation. \
# ### Question: {Question} ### Options: {Options}. ### Answer:"

single_choice_prefix = "This is a mammography-related medical question with several options, only one of which is correct. \
Select the correct answer and respond with just the chosen option, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
multiple_choice_prefix = "This is a mammography-related medical question with several options, one or more of which may be correct. \
Select the correct answers and respond with only the chosen options, without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"
yesorno_prefix="This is a mammography-related medical question with 'Yes' or 'No' options. \
Respond with only 'Yes' or 'No' without any further explanation. \
### Question: {Question} ### Options: {Options}. ### Answer:"

# cs_prefix="This is a mammography-related medical question. I want to assess your certainty. \
# Please respond with only 'Yes' or 'No' without any further explanation. \
# ### Question: {Question} ### Answer:"
# cs_question_single="Is the {Question_topic} of this mammogram {answer}?"
# cs_question_multiple="Does this mammogram show the presence of {answer}?"
# cs_question_multiple_normal="Is this mammogram normal?"
# cs_question_yesorno="Is {Question_topic} {answer} present in this mammogram?"

def build_prompt(sample,score_type):
    question_topic = sample['Question topic']
    question_type = sample['Question type']
    question = sample['Question']
    options = sample['Options']
    answer = sample['Answer']

    if score_type=='question_answering_score':
        if question_type == 'single choice':
            hint = Hint.get(question_topic, "")
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            # formatted_options=", ".join(options)
            prompt = single_choice_prefix.format(Question=question, Options=formatted_options, Hint=hint)
        elif question_type == 'yes/no':
            hint = Hint.get(question_topic, "")
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            prompt = yesorno_prefix.format(Question=question, Options=formatted_options, Hint=hint)
        else:
            hint = Hint.get(question_topic, "")
            random.shuffle(options)
            shuffled_options=[f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
            formatted_options = ", ".join(shuffled_options)
            # formatted_options=", ".join(options)
            prompt = multiple_choice_prefix.format(Question=question, Options=formatted_options, Hint=hint)
        
    # elif score_type=='certrain_score':
    # else:
    #     if question_type == 'single choice':
    #         hint = Hint.get(question_topic, "")
    #         # formatted_options = "A: Yes, B: No"
    #         prompt = cs_prefix.format(Question=cs_question_single.format(Question_topic=question_topic,answer=answer), Hint=hint)
    #     elif question_type == 'yes/no':
    #         hint = Hint.get(question_topic, "")
    #         # formatted_options = "A: Yes, B: No"
    #         prompt = cs_prefix.format(Question=cs_question_yesorno.format(Question_topic=question_topic,answer='' if answer=='Yes' else 'not'), Hint=hint)
    #     else:
    #         hint = Hint.get(question_topic, "")
    #         answer=', '.join(answer)
    #         # formatted_options = "A: Yes, B: No"
    #         if answer!='Normal':
    #             prompt = cs_prefix.format(Question=cs_question_multiple.format(answer=answer), Hint=hint)
    #         else:
    #             prompt = cs_prefix.format(Question=cs_question_multiple_normal, Hint=hint)
            
    return prompt

# def process_json_file(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     for sample in data:
#         prompt = build_prompt(sample)
#         print(f"ID: {sample['ID']}")
#         print(f"Prompt: {prompt}")
#         print(f"Answer: {sample['Answer']}")
#         print("-" * 40)

# # 处理 JSON 文件
# process_json_file('data.json')