# import difflib
import numpy as np
import re
# from fuzzywuzzy import process
import difflib
def uni_option(x,label_options):
    if x in label_options:
        return x
    else:
        for label_option in label_options:
            label, option=label_option.split(': ')
            if x==label or x==option:
                return label_option
def question_answering_score(prediction,shuffled_options,target,question_type):
    labels = [option.split(":")[0].strip() for option in shuffled_options]
    options = [option.split(":")[1].strip() for option in shuffled_options]
    formatted_options = labels + shuffled_options + options
    if question_type!='multiple choice':

        # 计算相似度
        similarities = []
        for option in formatted_options:
            similarity = difflib.SequenceMatcher(None, prediction.lower(), option.lower()).ratio()
            similarities.append((option, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 找出并列第一的选项
        max_score = similarities[0][1]
        top_options = [option for option, score in similarities if score == max_score]

        # 比较并列第一的选项
        top_options=[uni_option(top_option,shuffled_options) for top_option in top_options]
        if len(top_options) > 1:
            # 检查是否所有并列第一的选项都相同
            same_option = len(set(top_options))==1
            if same_option:
                # 选择 top_options 中的第一个选项
                best_option = top_options[0]
            else:
                return 0.0
        else:
            best_option = top_options[0]
        target=uni_option(target,shuffled_options)
        if best_option == target:
            return 1.0
        else:
            return 0.0
    else:
        keywords = re.findall(r'\b\w+\b', prediction.lower())
        
        selected_options = []
        for option in formatted_options:
            option_keywords = re.findall(r'\b\w+\b', option.lower())
            match_count = sum(1 for keyword in keywords if keyword in option_keywords)
            if match_count > 0:
                selected_options.append(option)
        selected_options=set([uni_option(selected_option,shuffled_options) for selected_option in selected_options])
        
        target=set([uni_option(_,shuffled_options) for _ in target])
        # 比较 final_options 和 label
        if selected_options == target:
            return 1.0
        else:
            return 0.0