import json
import numpy as np
from collections import defaultdict, Counter
import difflib
from fuzzywuzzy import process
import re
from scipy.stats import bootstrap, ttest_rel
from sklearn.metrics import f1_score
# Add the calculate_confidence_interval and calculate_p_value functions
def calculate_confidence_interval(scores, confidence_level=0.95):
    if len(scores) == 0:
        return (0.0, 0.0)
    res = bootstrap((np.array(scores),), np.mean, confidence_level=confidence_level)
    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    ci_low_percent = round(ci_low * 100, 2)
    ci_high_percent = round(ci_high * 100, 2)
    return (ci_low_percent, ci_high_percent)

def calculate_p_value(scores1, scores2):
    if len(scores1) == 0 or len(scores2) == 0:
        return 1.0
    if len(scores1) != len(scores2):
        return 1.0
    t_stat, p_value = ttest_rel(scores1, scores2)
    return p_value

# Modify the calculate_qas function to include CI and p-value calculations
def calculate_qas(eval_data, test_data, question_topic_info):
    scores_by_topic = defaultdict(list)
    target_by_topic = defaultdict(list)
    class_counts_by_topic = defaultdict(lambda: defaultdict(int))
    unique_categories = defaultdict(dict)

    scores_by_dataset = defaultdict(list)
    
    pred_labels_by_topic = defaultdict(list)
    gt_labels_by_topic = defaultdict(list)

    pred_labels_by_dataset = defaultdict(list)
    gt_labels_by_dataset = defaultdict(list)
    
    pred_abnormality_labels_by_dataset = defaultdict(list)
    gt_abnormality_labels_by_dataset = defaultdict(list)

    for topic, info in question_topic_info.items():
        categories = info[1]
        for idx, category in enumerate(categories):
            unique_categories[topic][category] = idx
        unique_categories[topic]["Make no choice"] = len(categories)

    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            question_topic = eval_sample['Question topic']
            question_type = eval_sample['Question type']
            category = eval_sample['Answer']
            dataset = eval_sample['Dataset']

            # if isinstance(category, list):
            #     category = tuple(category)

            if question_topic=="Abnormality":
                prediction=test_sample['qas_answer'].split(', ')
            else:
                prediction=test_sample['qas_answer']
            qas_score=float(prediction==category)
            scores_by_topic[question_topic].append(qas_score)
            scores_by_dataset[dataset].append(qas_score)

            if question_topic == "Abnormality":
                all_categories = ["Normal", "Calcification", "Mass", "Architectural distortion", 
                                "Asymmetry", "Miscellaneous", "Nipple retraction", 
                                "Suspicious lymph node", "Skin thickening", "Skin retraction"]
                cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
                num_classes = len(all_categories)
                gt_vec = [0]*num_classes
                for ans in category:
                    gt_vec[cat_to_idx[ans]] = 1
                gt_abnormality_labels_by_dataset[dataset].append(gt_vec)

                # Pred multi-hot
                # if isinstance(pred_label, list):  # extract_best_option 给出的预测可能是 list
                #     pred_ans = pred_label
                # else:
                #     pred_ans = [pred_label] if pred_label in cat_to_idx else []
                pred_vec = [0]*len(all_categories)
                for ans in prediction:
                    if ans in cat_to_idx:
                        pred_vec[cat_to_idx[ans]] = 1
                pred_abnormality_labels_by_dataset[dataset].append(pred_vec)
                continue
            else:
                gt_labels_by_dataset[dataset].append(unique_categories[question_topic][category])
                pred_labels_by_dataset[dataset].append(unique_categories[question_topic].get(prediction, -1))
            gt_labels_by_topic[question_topic].append(unique_categories[question_topic][category])
            pred_labels_by_topic[question_topic].append(unique_categories[question_topic].get(prediction, -1))
            target_by_topic[question_topic].append(unique_categories[question_topic][category])
            class_counts_by_topic[question_topic][category] += 1
        else:
            scores_by_topic[question_topic].append(0.0)
            scores_by_dataset[dataset].append(0.0)
            if question_topic == "Abnormality":
                continue
            target_by_topic[question_topic].append(-1)

    weighted_accuracy_by_topic = {}
    simple_accuracy_by_topic = {}
    confidence_intervals_by_topic = {}
    p_values_by_topic = {}

    for question_topic, scores in scores_by_topic.items():
        if question_topic == "Abnormality":
            total_score = np.sum(scores)
            total_count = len(scores)
            simple_accuracy = total_score / total_count if total_count > 0 else 0.0
            weighted_accuracy_by_topic[question_topic] = simple_accuracy
            simple_accuracy_by_topic[question_topic] = simple_accuracy
            confidence_intervals_by_topic[question_topic] = calculate_confidence_interval(scores)
            continue

        target = np.array(target_by_topic[question_topic])
        valid_indices = target != -1
        scores = np.array(scores)[valid_indices]
        target = target[valid_indices]

        categories = question_topic_info[question_topic][1]
        class_counts = np.array([class_counts_by_topic[question_topic].get(category, 0) 
                                 for category in categories])

        n = len(scores)
        if n == 0 or np.sum(class_counts) == 0:
            weighted_accuracy_by_topic[question_topic] = 0.0
            simple_accuracy_by_topic[question_topic] = 0.0
            confidence_intervals_by_topic[question_topic] = (0.0, 0.0)
            continue

        weights = n / class_counts[target]
        weighted_accuracy = np.sum(weights * scores) / np.sum(weights)
        weighted_accuracy_by_topic[question_topic] = weighted_accuracy

        simple_accuracy = np.mean(scores)
        simple_accuracy_by_topic[question_topic] = simple_accuracy

        confidence_intervals_by_topic[question_topic] = calculate_confidence_interval(scores)

    simple_accuracy_by_dataset = {}
    confidence_intervals_by_dataset = {}
    for dataset, scores in scores_by_dataset.items():
        total_score = np.sum(scores)
        total_count = len(scores)
        simple_accuracy = total_score / total_count if total_count > 0 else 0.0
        simple_accuracy_by_dataset[dataset] = simple_accuracy
        confidence_intervals_by_dataset[dataset] = calculate_confidence_interval(scores)
        
    f1_by_topic = {}
    for topic in pred_labels_by_topic:
        gt = np.array(gt_labels_by_topic[topic])
        pred = np.array(pred_labels_by_topic[topic])
        valid_idx = pred != -1 
        if np.any(valid_idx):
            f1_by_topic[topic] = f1_score(gt[valid_idx], pred[valid_idx], average="macro", zero_division=0)
        else:
            f1_by_topic[topic] = 0.0

    f1_by_dataset = {}
    all_datasets = set(gt_labels_by_dataset.keys()) | set(gt_abnormality_labels_by_dataset.keys())
    for dataset in all_datasets:
        scores = []
        counts = []

      
        if dataset in gt_labels_by_dataset:
            y_true = np.array(gt_labels_by_dataset[dataset])
            y_pred = np.array(pred_labels_by_dataset[dataset])
            valid_idx = y_pred != -1
            y_true = y_true[valid_idx]
            y_pred = y_pred[valid_idx]
            if len(y_true) > 0:
                scores.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
                counts.append(len(y_true))

   
        if dataset in gt_abnormality_labels_by_dataset:
            y_true = np.array(gt_abnormality_labels_by_dataset[dataset])
            y_pred = np.array(pred_abnormality_labels_by_dataset[dataset])
            if len(y_true) > 0:
                scores.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
                counts.append(len(y_true))

     
        if counts:
            f1_by_dataset[dataset] = np.average(scores, weights=counts)
        else:
            f1_by_dataset[dataset] = 0.0

    return (weighted_accuracy_by_topic, simple_accuracy_by_topic, confidence_intervals_by_topic, 
            simple_accuracy_by_dataset, confidence_intervals_by_dataset,
            f1_by_topic, f1_by_dataset) 

# Modify the calculate_pathology_qas function to include CI and p-value calculations
def calculate_pathology_qas(eval_data, test_data, question_topic_info):
    scores_finding = defaultdict(list)
    scores_non_finding = defaultdict(list)
    target_finding = defaultdict(list)
    target_non_finding = defaultdict(list)
    class_counts_finding = defaultdict(lambda: defaultdict(int))
    class_counts_non_finding = defaultdict(lambda: defaultdict(int))
    unique_categories = defaultdict(dict)
    
    pred_labels_by_finding = defaultdict(list)
    gt_labels_by_finding = defaultdict(list)

    pred_labels_by_non_finding = defaultdict(list)
    gt_labels_by_non_finding = defaultdict(list)

    for topic, info in question_topic_info.items():
        categories = info[1]
        for idx, category in enumerate(categories):
            unique_categories[topic][category] = idx

    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            question_topic = eval_sample['Question topic']
            question_type = eval_sample['Question type']
            category = eval_sample['Answer']
            dataset = eval_sample['Dataset']

            if question_topic in ['Pathology']:
                if isinstance(category, list):
                    category = tuple(category)

                prediction=test_sample['qas_answer']
                qas_score=float(prediction==category)

                if dataset.endswith("finding"):
                    scores_finding[question_topic].append(qas_score)
                    target_finding[question_topic].append(unique_categories[question_topic][category])
                    gt_labels_by_finding[question_topic].append(unique_categories[question_topic][category])
                    pred_labels_by_finding[question_topic].append(unique_categories[question_topic].get(prediction, -1))
                    class_counts_finding[question_topic][category] += 1
                else:
                    scores_non_finding[question_topic].append(qas_score)
                    target_non_finding[question_topic].append(unique_categories[question_topic][category])
                    gt_labels_by_non_finding[question_topic].append(unique_categories[question_topic][category])
                    pred_labels_by_non_finding[question_topic].append(unique_categories[question_topic].get(prediction, -1))
                    class_counts_non_finding[question_topic][category] += 1

    weighted_accuracy_finding = {}
    simple_accuracy_finding = {}
    confidence_intervals_finding = {}
    for question_topic, scores in scores_finding.items():
        target = np.array(target_finding[question_topic])
        valid_indices = target != -1
        scores = np.array(scores)[valid_indices]
        target = target[valid_indices]

        categories = question_topic_info[question_topic][1]
        class_counts = np.array([class_counts_finding[question_topic].get(category, 0) 
                                 for category in categories])

        n = len(scores)
        if n == 0 or np.sum(class_counts) == 0:
            weighted_accuracy_finding[question_topic] = 0.0
            simple_accuracy_finding[question_topic] = 0.0
            confidence_intervals_finding[question_topic] = (0.0, 0.0)
            continue

        weights = n / class_counts[target]
        weighted_accuracy = np.sum(weights * scores) / np.sum(weights)
        weighted_accuracy_finding[question_topic] = weighted_accuracy

        simple_accuracy = np.mean(scores)
        simple_accuracy_finding[question_topic] = simple_accuracy

        confidence_intervals_finding[question_topic] = calculate_confidence_interval(scores)

    weighted_accuracy_non_finding = {}
    simple_accuracy_non_finding = {}
    confidence_intervals_non_finding = {}
    for question_topic, scores in scores_non_finding.items():
        target = np.array(target_non_finding[question_topic])
        valid_indices = target != -1
        scores = np.array(scores)[valid_indices]
        target = target[valid_indices]

        categories = question_topic_info[question_topic][1]
        class_counts = np.array([class_counts_non_finding[question_topic].get(category, 0) 
                                 for category in categories])

        n = len(scores)
        if n == 0 or np.sum(class_counts) == 0:
            weighted_accuracy_non_finding[question_topic] = 0.0
            simple_accuracy_non_finding[question_topic] = 0.0
            confidence_intervals_non_finding[question_topic] = (0.0, 0.0)
            continue

        weights = n / class_counts[target]
        weighted_accuracy = np.sum(weights * scores) / np.sum(weights)
        weighted_accuracy_non_finding[question_topic] = weighted_accuracy

        simple_accuracy = np.mean(scores)
        simple_accuracy_non_finding[question_topic] = simple_accuracy

        confidence_intervals_non_finding[question_topic] = calculate_confidence_interval(scores)

    f1_by_finding = {}
    for topic in gt_labels_by_finding:
        f1_by_finding[topic] = f1_score(gt_labels_by_finding[topic], pred_labels_by_finding[topic], average="macro")

    f1_by_non_finding = {}
    for topic in gt_labels_by_non_finding:
        f1_by_non_finding[topic] = f1_score(gt_labels_by_non_finding[topic], pred_labels_by_non_finding[topic], average="macro")

    return (weighted_accuracy_finding, simple_accuracy_finding, confidence_intervals_finding,f1_by_finding), \
           (weighted_accuracy_non_finding, simple_accuracy_non_finding, confidence_intervals_non_finding,f1_by_non_finding)

# Modify the calculate_abnormality_qas function to include CI and p-value calculations
def calculate_abnormality_qas(eval_data, test_data, question_topic):
    finding_counter = Counter()
    non_finding_counter = Counter()
    
    all_categories = ["Normal", "Calcification", "Mass", "Architectural distortion", 
                      "Asymmetry", "Miscellaneous", "Nipple retraction", 
                      "Suspicious lymph node", "Skin thickening", "Skin retraction"]

    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}

    gt_labels_finding = []
    pred_labels_finding = []
    gt_labels_non_finding = []
    pred_labels_non_finding = []

    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            dataset = eval_sample['Dataset']
            if dataset.endswith("finding"):
                if eval_sample["Question topic"] == question_topic:
                    answer_group = tuple(sorted(eval_sample["Answer"]))
                    finding_counter[answer_group] += 1
            else:
                if eval_sample["Question topic"] == question_topic:
                    answer_group = tuple(sorted(eval_sample["Answer"]))
                    non_finding_counter[answer_group] += 1

    finding_categories = list(finding_counter.keys())
    finding_class_counts = np.array([finding_counter.get(category, 0) for category in finding_categories])

    finding_scores = []
    finding_targets = []
    finding_n = 0

    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            if eval_sample["Question topic"] == question_topic and eval_sample['Dataset'].endswith("finding"):
                gt_answers = tuple(sorted(eval_sample["Answer"])) 
                pred = test_data.get(idx, {}).get("qas_answer", "")
                pred_answers = tuple(sorted(pred.split(", ")))
                
                finding_targets.append(finding_categories.index(gt_answers))
                finding_scores.append(1 if pred_answers == gt_answers else 0)
                finding_n += 1
                
                gt_vec = [0]*len(all_categories)
                for ans in eval_sample["Answer"]:
                    gt_vec[cat_to_idx[ans]] = 1
                gt_labels_finding.append(gt_vec)


                pred_vec = [0]*len(all_categories)
                for ans in pred_answers:
                    if ans in cat_to_idx:
                        pred_vec[cat_to_idx[ans]] = 1
                pred_labels_finding.append(pred_vec)

    finding_scores = np.array(finding_scores)
    finding_targets = np.array(finding_targets)

    if finding_n == 0 or np.sum(finding_class_counts) == 0:
        finding_weighted_accuracy = 0.0
        finding_simple_accuracy = 0.0
        finding_confidence_interval = (0.0, 0.0)
        f1_finding = 0.0
    else:
        weights = finding_n / finding_class_counts[finding_targets]
        finding_weighted_accuracy = np.sum(weights * finding_scores) / np.sum(weights)
        finding_simple_accuracy = np.mean(finding_scores)
        finding_confidence_interval = calculate_confidence_interval(finding_scores)
        f1_finding = f1_score(gt_labels_finding, pred_labels_finding, average="macro", zero_division=0)

    non_finding_categories = list(non_finding_counter.keys())
    non_finding_class_counts = np.array([non_finding_counter.get(category, 0) for category in non_finding_categories])

    non_finding_scores = []
    non_finding_targets = []
    non_finding_n = 0

    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            if eval_sample["Question topic"] == question_topic and eval_sample['Dataset'].endswith("breast"):
                gt_answers = tuple(sorted(eval_sample["Answer"]))
                pred = test_data.get(idx, {}).get("qas_answer", "")  
                pred_answers = tuple(sorted(pred.split(", ")))

                non_finding_targets.append(non_finding_categories.index(gt_answers))
                non_finding_scores.append(1 if pred_answers == gt_answers else 0)
                non_finding_n += 1
                
                gt_vec = [0]*len(all_categories)
                for ans in eval_sample["Answer"]:
                    gt_vec[cat_to_idx[ans]] = 1
                gt_labels_non_finding.append(gt_vec)

                # if isinstance(qas_score, tuple):
                #     pred_ans = qas_score[1] if isinstance(qas_score[1], list) else [qas_score[1]]
                # else:
                #     pred_ans = eval_sample["Answer"] if qas_score == 1.0 else []
                pred_vec = [0]*len(all_categories)
                for ans in pred_answers:
                    if ans in cat_to_idx:
                        pred_vec[cat_to_idx[ans]] = 1
                pred_labels_non_finding.append(pred_vec)

    non_finding_scores = np.array(non_finding_scores)
    non_finding_targets = np.array(non_finding_targets)

    if non_finding_n == 0 or np.sum(non_finding_class_counts) == 0:
        non_finding_weighted_accuracy = 0.0
        non_finding_simple_accuracy = 0.0
        non_finding_confidence_interval = (0.0, 0.0)
        f1_non_finding = 0.0
    else:
        weights = non_finding_n / non_finding_class_counts[non_finding_targets]
        non_finding_weighted_accuracy = np.sum(weights * non_finding_scores) / np.sum(weights)
        non_finding_simple_accuracy = np.mean(non_finding_scores)
        non_finding_confidence_interval = calculate_confidence_interval(non_finding_scores)
        f1_non_finding = f1_score(gt_labels_non_finding, pred_labels_non_finding, average="macro", zero_division=0)

    return (finding_weighted_accuracy, finding_simple_accuracy, finding_confidence_interval, f1_finding), \
           (non_finding_weighted_accuracy, non_finding_simple_accuracy, non_finding_confidence_interval, f1_non_finding)

# Modify the format_qas_cs_output function to display CI and p-value results
def format_qas_cs_output(eval_data, test_data, question_topic_info):
    dataset_order = [
        "MIAS-breast", "DMID-breast", "BMCD", "INbreast", "KAU-BCMD", 
        "VinDr-Mammo-breast", "CDD-CESM", "CBIS-DDSM-breast", "CSAW-M", 
        "RSNA", "DMID-finding", "MIAS-finding", "VinDr-Mammo-finding", "CBIS-DDSM-finding"
    ]
    question_topic_order = [
        "Background tissue", "Subtlety", "Masking potential", 
        "Bi-Rads", "ACR", "View", "Laterality"
    ]


    (weighted_acc_by_topic, simple_acc_by_topic, confidence_intervals_by_topic, 
     simple_acc_by_dataset, confidence_intervals_by_dataset,
            f1_by_topic, f1_by_dataset) = calculate_qas(eval_data, test_data, question_topic_info)

    print("\nF1 by Dataset:")
    for dataset, value in f1_by_dataset.items():
        print(dataset)
    for dataset in dataset_order:
        print(f"Dataset {dataset}: F1 = {f1_by_dataset[dataset]*100:.2f}%")


    print("\nF1 by Question Topic:")
    for topic in question_topic_order:
        print(f"Question Topic {topic}: F1 = {f1_by_topic[topic]*100:.2f}%")
    

    print("\nSimple Accuracy by Dataset:")
    for dataset in dataset_order:
        if dataset in simple_acc_by_dataset:
            acc_score = simple_acc_by_dataset[dataset] * 100
            ci_low, ci_high = confidence_intervals_by_dataset[dataset]
            print(f"Dataset {dataset}: Simple Accuracy = {acc_score:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")
        else:
            print(f"Dataset {dataset}: Simple Accuracy = N/A")

    print("\nWeighted Accuracy by Question Topic:")
    for topic in question_topic_order:
        if topic in weighted_acc_by_topic:
            acc_score = weighted_acc_by_topic[topic] * 100
            ci_low, ci_high = confidence_intervals_by_topic[topic]
            print(f"Question Topic {topic}: Weighted Accuracy = {acc_score:.2f}%")
        else:
            print(f"Question Topic {topic}: Weighted Accuracy = N/A")

    print("\nSimple Accuracy by Question Topic:")
    for topic in question_topic_order:
        if topic in simple_acc_by_topic:
            acc_score = simple_acc_by_topic[topic] * 100
            ci_low, ci_high = confidence_intervals_by_topic[topic]
            print(f"Question Topic {topic}: Simple Accuracy = {acc_score:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")
        else:
            print(f"Question Topic {topic}: Simple Accuracy = N/A")

    ((pathology_weighted_finding, pathology_simple_finding, pathology_ci_finding,f1_by_finding), 
     (pathology_weighted_non_finding, pathology_simple_non_finding, pathology_ci_non_finding,f1_by_non_finding)) = calculate_pathology_qas(eval_data, test_data, question_topic_info)
    
    print("\nF1 by Dataset:")
    for topic, f1 in f1_by_non_finding.items():
        print(f"Pathology (breast): F1 = {f1*100:.2f}%")

    print("\nF1 by Question Topic:")
    for topic, f1 in f1_by_finding.items():
        print(f"Pathology (finding): F1 = {f1*100:.2f}%")
        
    print("\nPathology QAS by Breast:")
    for topic in pathology_weighted_non_finding:
        weighted_acc = pathology_weighted_non_finding[topic] * 100
        simple_acc = pathology_simple_non_finding[topic] * 100
        ci_low, ci_high = pathology_ci_non_finding[topic]
        print(f"Pathology (breast): Weighted Accuracy = {weighted_acc:.2f}%, Simple Accuracy = {simple_acc:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")
    print("\nPathology QAS by Finding:")
    for topic in pathology_weighted_finding:
        weighted_acc = pathology_weighted_finding[topic] * 100
        simple_acc = pathology_simple_finding[topic] * 100
        ci_low, ci_high = pathology_ci_finding[topic]
        print(f"Pathology (finding): Weighted Accuracy = {weighted_acc:.2f}%, Simple Accuracy = {simple_acc:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")

    ((abnormality_weighted_finding, abnormality_simple_finding, abnormality_ci_finding, f1_finding), 
     (abnormality_weighted_non_finding, abnormality_simple_non_finding, abnormality_ci_non_finding, f1_non_finding)) = calculate_abnormality_qas(eval_data, test_data, "Abnormality")
    
    print(f"\nAbnormality (breast): F1 = {f1_non_finding*100:.2f}%")
    print(f"\nAbnormality (finding): F1 = {f1_finding*100:.2f}%")
    print("\nAbnormality QAS by Breast:")
    print(f"Abnormality (breast): Weighted Accuracy = {abnormality_weighted_non_finding * 100:.2f}%, Simple Accuracy = {abnormality_simple_non_finding * 100:.2f}%, CI = ({abnormality_ci_non_finding[0]:.2f}%, {abnormality_ci_non_finding[1]:.2f}%)")
    print("\nAbnormality QAS by Finding:")
    print(f"Abnormality (finding): Weighted Accuracy = {abnormality_weighted_finding * 100:.2f}%, Simple Accuracy = {abnormality_simple_finding * 100:.2f}%, CI = ({abnormality_ci_finding[0]:.2f}%, {abnormality_ci_finding[1]:.2f}%)")


# Example Usage
if __name__ == "__main__":
    with open('/home/jiayi/MammoVQA/Benchmark/MammoVQA-Image-Bench.json', 'r') as f:
        eval_data = json.load(f)

    method='DiNOv2'
    with open(f'/home/jiayi/MammoVQA/Result/{method}.json', 'r') as f:
        test_data = json.load(f)
    print(f'-------{method}--------')

    question_topic_info = {
        "Laterality": ["single choice", ["Right", "Left"]],
        "View": ["single choice", ["MLO", "CC"]],
        "ACR": ["single choice", ["Level A", "Level B", "Level C", "Level D"]],
        "Bi-Rads": ["single choice", ["Bi-Rads 0", "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3", "Bi-Rads 4", "Bi-Rads 5", "Bi-Rads 6"]],
        "Background tissue": ["single choice", ["Fatty-glandular", "Fatty", "Dense-glandular"]],
        "Subtlety": ["single choice", ["Normal", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]],
        "Masking potential": ["single choice", ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7", "Level 8"]],
        "Pathology": ["single choice", ["Normal", "Malignant", "Benign"]],
        "Abnormality": ["multiple choice", ["Normal", "Calcification", "Mass", "Architectural distortion", "Asymmetry", "Miscellaneous", "Nipple retraction", "Suspicious lymph node", "Skin thickening", "Skin retraction"]]
    }

    format_qas_cs_output(eval_data, test_data, question_topic_info)
