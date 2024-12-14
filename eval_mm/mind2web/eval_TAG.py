import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
from PIL import Image
import numpy as np
import copy

from core_utils.eval_main import EvalMain
from prompt_tag import TYPE_VALUE_NAME, INTERESTED_KEY, history_template, action_space, type_statement, query_prompt
from m2w_utils import action2step, action2id, resize_image, transform_gt_bbox, calculate_f1, find_element_content_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument("--num_history", type=int, default=4, help="number of history steps for agent")
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--vis_flag", action='store_true', help="if vis mid-results")
    # params for TAG
    parser.add_argument("--filter_thresold", type=float, default=0.5)
    parser.add_argument("--topk_layer_head", type=int, default=5, help="select topk image-attn-weight from all layer heads")
    args = parser.parse_args()
    return args

args = parse_args()


tag_params = {
        "filter_thresold": args.filter_thresold,
        "topk_layer_head": args.topk_layer_head,
        "vis_flag": args.vis_flag
    }
eval_main = EvalMain(mllm_path=args.mllm_path, tag_param_dict=tag_params, query_prompt=query_prompt, vis_flag=args.vis_flag)


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
        if "bbox" not in step:
            print("action not found")
            continue
        
        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        save_path = os.path.join(args.save_dir, f"{goal_folder}/step{j}_{filename[:-4]}")
        cur_json_save_path = os.path.join(save_path, 'step_result.json')

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
        action_step, cur_action_step = action2step(step)
        try:
            if cur_action_step[INTERESTED_KEY]!='':
                previous_actions.append(action_step)
        except:
            pass
        
        if os.path.exists(cur_json_save_path):
            continue
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
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

        # get generation
        prompt = query_prompt.format(goal=goal, history=history_info, action_space=action_space, interested_key=INTERESTED_KEY, type_statement=type_statement)
        msgs = [{'role': 'user', 'content': prompt}]
        response, attention_outputs = eval_main.model.chat(image, msgs, eval_main.tokenizer, query_text_attention=True, sampling=False, temperature=0., num_beams=1, output_attentions=True, return_dict_in_generate=True, output_attentions_generation=True)

        gen_query_idx, response_parse = find_element_content_indices(response, eval_main.tokenizer)
        print(f"\n\n{website}-Step{j}: Filename: {filename}\nQuery:\n{prompt}\nResponse:\n{response}\nGround Truth:\n{action_step}\nLocation Words: {response_parse[INTERESTED_KEY]}\n")
        attention_outputs['gen_query_idx']=gen_query_idx
        attention_outputs['gt_bboxes'] = [bbox_trans]
        attention_outputs["original_image"] = image
        tag_output = eval_main.forward(attention_outputs, save_path)
        tag_output['prompt'] = prompt
        tag_output["response_parse"] = response_parse
        tag_output['gt_action'] = action_step
        # get final decision
        tag_res_from_prompt_acc = tag_output['tag_res_from_prompt']["final_decision"]['click_point_acc']
        click_point_acc = tag_res_from_prompt_acc
        if len(tag_output['tag_res_from_gen'])>1:
            tag_res_from_gen_acc = tag_output['tag_res_from_gen']["final_decision"]['click_point_acc']
            click_point_acc = tag_res_from_gen_acc
        else:
            tag_res_from_gen_acc = False

        gt_action_id = action2id(cur_action_step["action_type"])
        step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                       "Op_match": False, "Ele_match": False, "Op_F1": [0, gt_action_id], "Ele_match_prompt": tag_res_from_prompt_acc}
        try:
            response_action_id = action2id(response_parse["action_type"])
            if response_action_id == gt_action_id:
                step_result["Op_match"] = True

            if click_point_acc:
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
            print("format wrong")

        tag_output['step_result'] = step_result
        
        with open(cur_json_save_path, "w") as f:
            json.dump(tag_output, f, indent=4, ensure_ascii=False)
        print(f"{step_result}\n"+"-"*150)

        results_actions.append(step_result)

    results.append(results_actions)


# calculate metrics
num_step = 0
num_episode = 0
num_op = 0
num_ele = 0
num_ele_prompt = 0
op_f1 = {4: [], 2: [], 3: []}
macro_ele_acc = {}
macro_ele_acc_prompt = {} # do tag on prompt
macro_step_acc = {}
macro_step_acc_prompt = {}
macro_action_f1 = {}
num_step_success = 0
num_step_success_prompt = 0
num_episode_success = 0
num_episode_success_prompt = 0
for i, item in enumerate(results):
    macro_ele_acc[i] = []
    macro_ele_acc_prompt[i] = []
    macro_step_acc[i] = []
    macro_step_acc_prompt[i] = []
    macro_action_f1[i] = []
    num_episode += 1
    episode_success = True
    episode_success_prompt = True
    for step_result in item:
        num_step += 1

        if step_result["Op_match"]:
            num_op += 1

        if step_result["Ele_match"]:
            num_ele += 1
            macro_ele_acc[i].append(1)
        else:
            macro_ele_acc[i].append(0)
        
        if step_result["Ele_match_prompt"]:
            num_ele_prompt += 1
            macro_ele_acc_prompt[i].append(1)
        else:
            macro_ele_acc_prompt[i].append(0)

        if step_result["Op_F1"][1] in op_f1:
            op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
        macro_action_f1[i].append(step_result["Op_F1"][0])

        if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
            num_step_success += 1
            macro_step_acc[i].append(1)
        else:
            macro_step_acc[i].append(0)
            episode_success = False

        if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match_prompt"]:
            num_step_success_prompt += 1
            macro_step_acc_prompt[i].append(1)
        else:
            macro_step_acc_prompt[i].append(0)
            episode_success_prompt = False


    if episode_success:
        num_episode_success += 1
    
    if episode_success_prompt:
        num_episode_success_prompt += 1

marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])

print(args.task + ": Operation F1: " + str(marco_op_f1))
print(args.task + ": Element Acc: " + str(num_ele / num_step))
print(args.task + ": Element Acc_prompt: " + str(num_ele_prompt / num_step))
print(args.task + ": Step Success: " + str(num_step_success / num_step))
print(args.task + ": Episode Success: " + str(num_episode_success / num_episode))
print(args.task + ": Episode Success Prompt: " + str(num_episode_success_prompt / num_episode))
print(args.task + ": Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
macro_ele_acc_prompt = np.mean([np.mean(x) for x in macro_ele_acc_prompt.values()])
macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
macro_step_acc_prompt = np.mean([np.mean(x) for x in macro_step_acc_prompt.values()])
macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])
print(args.task + ": Macro Ele Acc: " + str(macro_ele_acc))
print(args.task + ": Macro Ele Acc Prompt: " + str(macro_ele_acc_prompt))
print(args.task + ": Macro Op F1: " + str(macro_action_f1))
print(args.task + ": Macro Step SR: " + str(macro_step_acc))
print(args.task + ": Macro Step SR Prompt: " + str(macro_step_acc_prompt))