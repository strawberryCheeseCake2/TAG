# evaluation on mind2web
import os
import torch
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
import ast
import argparse
from PIL import Image
import numpy as np
import copy

from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper
from prompt_tag import TYPENAME, TYPE_VALUE_NAME, INTERESTED_KEY
from m2w_utils import AVAILABLE_ACTION_TYPE, action2id, parse_choice, resize_image, transform_gt_bbox, calculate_f1, check_available, coarse_find_action_type, coarse_find_value
from prompt_default import history_template, action_space, type_statement, prompt_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--num_history", type=int, default=4, help="number of history steps for agent")
    args = parser.parse_args()
    return args

args = parse_args()

def bbox_pred2gt(bbox, img_size):
    img_w, img_h = img_size
    resize_ratio_w = img_w / 1000
    resize_ratio_h = img_h / 1000
    return [int(bbox[0]*resize_ratio_w), int(bbox[1]*resize_ratio_h), int(bbox[2]*resize_ratio_w), int(bbox[3]*resize_ratio_h)]

def bbox_gt2pred(bbox, img_size):
    img_w, img_h = img_size
    resize_ratio_w = 1000 / img_w
    resize_ratio_h = 1000 / img_h
    return [int(bbox[0]*resize_ratio_w), int(bbox[1]*resize_ratio_h), int(bbox[2]*resize_ratio_w), int(bbox[3]*resize_ratio_h)]

def action2step(action, bbox):
    action_type = action["operation"]["original_op"]
    assert action_type in AVAILABLE_ACTION_TYPE

    if len(action['pos_candidates'])>0:
        choice = action['pos_candidates'][0]['choice']
        choice_parse = parse_choice(choice)
    else:
        choice_parse = ""

    position_str = f"<box>{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}</box>"

    action_step_dict = dict()
    action_step_dict['action_type'] = action_type.lower()
    action_step_dict[INTERESTED_KEY] = choice_parse
    action_step_dict['position'] = position_str
    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        # history_action_name = 'click'
        history_action_name = action_type.lower()
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\", \"position\": \"{}\"}}".format(history_action_name, INTERESTED_KEY, choice_parse, position_str)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        # history_action_name = 'click'
        history_action_name = action_type.lower()
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\", \"position\": \"{}\"}}".format(history_action_name, INTERESTED_KEY, select_value, position_str)
        action_step_dict[INTERESTED_KEY] = select_value
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": \"{}\", \"{}\": \"{}\", \"position\": \"{}\", \"{}\": \"{}\"}}".format(TYPENAME, INTERESTED_KEY, choice_parse, position_str ,TYPE_VALUE_NAME, typed_text)
        action_step_dict['action_type'] = TYPENAME
        action_step_dict[TYPE_VALUE_NAME] = typed_text

    return action_step.replace('"', "'"), action_step_dict

def parse_response(response, input_img_size):
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, response)
    while len(matches)>1 and not check_available(matches[-1]):
        # if more than 1 actions are generated, filter out actions with not available action_type
        matches = matches[:-1]
    if len(matches)>=1:
        json_string = matches[-1]
    else:
        json_string = response
    
    # parse bbox
    bbox_pattern = r"(?:<box>|\[)(\d+) (\d+) (\d+) (\d+)"
    res_bbox = []
    matches = re.findall(bbox_pattern, json_string)
    if len(matches)>0:
        for match in matches:
            res_bbox.append([
                max(int(match[0]), 0),
                max(int(match[1]), 0),
                max(int(match[2]), 0),
                max(int(match[3]), 0)
            ])
        pred_bbox = res_bbox[0]
    else:
        pred_bbox = [0, 0, 0, 0]
    pred_bbox = bbox_pred2gt(pred_bbox, input_img_size)

    # parse action type
    try:
        json_string_eval = json_string.replace('\n', ' ')
        json_string_eval = json_string_eval.replace("'s ", "@@@s ")
        json_string_eval = json_string_eval.replace("'t ", "@@@t ")
        data = ast.literal_eval(json_string_eval)
    except:
        print("*"*100+ f"\nFail to parse: {json_string}\n"+"*"*100)
        data = dict()
        data['action_type'] = coarse_find_action_type(json_string)
        if data['action_type'] == TYPENAME:
            data[TYPE_VALUE_NAME] = coarse_find_value(json_string)
    data['position'] = pred_bbox
    return data


model = MiniCPMV.from_pretrained(args.mllm_path, torch_dtype=torch.float16)
tokenizer = PreTrainedTokenizerFastWrapper.from_pretrained(args.mllm_path)
model = model.to(device='cuda')
model.eval()

mind2web_imgs_dir = os.path.join(args.data_dir, 'mind2web_images')
mind2web_test = json.load(open(os.path.join(args.data_dir, f'mind2web_data_test_' + args.task + '.json'), 'r'))

results = []
for episode in tqdm(mind2web_test):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    website = episode["website"]
    vis_goal = "_".join(goal.replace("/", "").split(' '))
    vis_goal = vis_goal[:100]
    goal_folder = f"{website}/{vis_goal}"
    previous_actions = []
    results_actions = []

    for j, step in enumerate(episode["actions"]):
        item_res = dict()
        if "bbox" not in step:
            print("action not found")
            continue
        
        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        save_path = os.path.join(args.save_dir, f"{goal_folder}/step{j}_{filename[:-4]}")
        cur_json_save_path = os.path.join(save_path, 'step_result.json')

        # load image and gt_bbox
        img_path = os.path.join(mind2web_imgs_dir, filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue
        original_image = Image.open(img_path)
        original_img_size = original_image.size
        image = resize_image(original_image, original_img_size)
        img_size = image.size
        bbox_trans = transform_gt_bbox(original_img_size, img_size, copy.deepcopy(step['bbox']))
        item_res['original_bbox'] = copy.copy(step['bbox'])
        item_res['bbox'] = copy.copy(bbox_trans)

        # get historty data
        if args.num_history>0:
            previous_step = ""
            for i, action in enumerate(previous_actions[-args.num_history:]):
                previous_step += '- Step' + str(i) + ': ' + action + " \n"
            if len(previous_step)>0:
                history_info = history_template.format(previous_actions=previous_step)
            else:
                history_info = ""
        else:
            history_info = ""
        # get current step ground truth
        bbox_trans_4_history = bbox_gt2pred(bbox_trans, img_size)
        action_step, cur_action_step = action2step(step, bbox_trans_4_history)
        try:
            if cur_action_step[INTERESTED_KEY]!='':
                previous_actions.append(action_step)
        except:
            pass
        
        if os.path.exists(cur_json_save_path):
            continue

        # get generation
        prompt = prompt_template.format(goal=goal, history=history_info, action_space=action_space, type_statement=type_statement)
        msgs = [{'role': 'user', 'content': prompt}]
        response = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, sampling=False, temperature=0.)
        print(f"\n\n{website}-Step{j}: Filename: {filename}\nQuery:\n{prompt}\nResponse:\n{response}\nGround Truth:\n{action_step}\n", flush=True)
        response_parse = parse_response(response, input_img_size=img_size)
        item_res['pred_bbox'] = response_parse['position']

        gt_action_id = action2id(cur_action_step["action_type"])
        step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                       "Op_match": False, "Ele_match": False, "Op_F1": [0, gt_action_id]}
        try:
            response_action_id = action2id(response_parse["action_type"])
            if response_action_id == gt_action_id:
                step_result["Op_match"] = True

            pred_bbox = response_parse["position"]
            click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            if (bbox_trans[0] <= click_point[0] <= bbox_trans[2]) and (bbox_trans[1] <= click_point[1] <= bbox_trans[3]):
                step_result["Ele_match"] = True

            pred_str = str(response_action_id)
            if response_action_id == 3:
                pred_str += ' '
                pred_str += response_parse[TYPE_VALUE_NAME].lower()
            elif gt_action_id == 2:
                pred_str += ' '
                pred_str += response_parse[INTERESTED_KEY].lower()
            ref_str = str(gt_action_id)
            if gt_action_id == 3:
                ref_str += ' '
                ref_str += cur_action_step[TYPE_VALUE_NAME].lower()
            elif gt_action_id == 2:
                ref_str += ' '
                ref_str += cur_action_step[INTERESTED_KEY].lower()

            op_f1 = calculate_f1(pred_str, ref_str)
            step_result["Op_F1"][0] = op_f1

        except:
            print("format wrong", flush=True)

        item_res['step_result'] = step_result
        if not os.path.exists(os.path.dirname(cur_json_save_path)):
            os.makedirs(os.path.dirname(cur_json_save_path))
        with open(cur_json_save_path, "w") as f:
            json.dump(item_res, f, indent=4, ensure_ascii=False)
        print(f"{step_result}\n"+"-"*150, flush=True)

        results_actions.append(step_result)

    results.append(results_actions)


# calculate metrics
num_step = 0
num_episode = 0
num_op = 0
num_ele = 0
op_f1 = {4: [], 2: [], 3: []}
macro_ele_acc = {}
macro_step_acc = {}
macro_action_f1 = {}
num_step_success = 0
num_episode_success = 0
for i, item in enumerate(results):
    macro_ele_acc[i] = []
    macro_step_acc[i] = []
    macro_action_f1[i] = []
    num_episode += 1
    episode_success = True
    for step_result in item:
        num_step += 1

        if step_result["Op_match"]:
            num_op += 1

        if step_result["Ele_match"]:
            num_ele += 1
            macro_ele_acc[i].append(1)
        else:
            macro_ele_acc[i].append(0)

        if step_result["Op_F1"][1] in op_f1:
            op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
        macro_action_f1[i].append(step_result["Op_F1"][0])

        if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
            num_step_success += 1
            macro_step_acc[i].append(1)
        else:
            macro_step_acc[i].append(0)
            episode_success = False

    if episode_success:
        num_episode_success += 1

marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])

print("Operation F1: " + str(marco_op_f1))
print("Element Acc: " + str(num_ele / num_step))
print("Step Success: " + str(num_step_success / num_step))
print("Episode Success: " + str(num_episode_success / num_episode))
print("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])
print("Macro Ele Acc: " + str(macro_ele_acc))
print("Macro Op F1: " + str(macro_action_f1))
print("Macro Step SR: " + str(macro_step_acc))