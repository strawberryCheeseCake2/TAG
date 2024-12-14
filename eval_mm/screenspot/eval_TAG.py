import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
from tqdm import tqdm
import copy
from collections import defaultdict
from PIL import Image

from core_utils.eval_main import EvalMain
from prompt_tag import query_prompt, interested_key, find_element_content_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")
    # params for TAG
    parser.add_argument("--filter_thresold", type=float, default=0.5)
    parser.add_argument("--topk_layer_head", type=int, default=10, help="select topk image-attn-weight from all layer heads")
    args = parser.parse_args()
    return args

args = parse_args()


def resize_image(original_image, original_size, max_size = 1344):
    W, H = original_size
    if W<=max_size and H<=max_size:
        return original_image
    if W>=H:
        W_new = max_size
        H_new = int(H/W * W_new)
    else:
        H_new = max_size
        W_new = int(W/H * H_new)
    return original_image.resize([W_new, H_new], Image.Resampling.BICUBIC)

def transform_gt_bbox(original_img_size, new_img_size, gt_dict):
    w_ratio = new_img_size[0]/original_img_size[0]
    h_ratio = new_img_size[1]/original_img_size[1]
    for query_text in list(gt_dict.keys()):
        for bbox in gt_dict[query_text]:
            bbox[0] *= w_ratio
            bbox[1] *= h_ratio
            bbox[2] *= w_ratio
            bbox[3] *= h_ratio
    return gt_dict


tag_params = {
        "filter_thresold": args.filter_thresold,
        "topk_layer_head": args.topk_layer_head,
        "vis_flag": args.vis_flag
    }
eval_main = EvalMain(mllm_path=args.mllm_path, tag_param_dict=tag_params, query_prompt=query_prompt, vis_flag=args.vis_flag)

tasks = ["desktop", "web", "mobile"]
for task in tasks:
    task_res = dict()
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))

    task_samples = defaultdict(dict)
    data_types = defaultdict(dict)
    for j, item in tqdm(enumerate(screenspot_data)):
        filename = item["img_filename"]
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        task_samples[filename][instruction] = [bbox]
        data_types[filename][instruction] = item["data_type"]

    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    for filename in list(task_samples.keys()):
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        original_image = Image.open(img_path).convert("RGB")
        original_img_size = original_image.size
        image = resize_image(original_image, original_img_size)
        img_size = image.size
        query_gts = transform_gt_bbox(original_img_size, img_size, copy.deepcopy(task_samples[filename]))
        
        task_res[filename] = dict()
        task_res[filename]['original_img_size'] = original_img_size
        task_res[filename]['input_image_size'] = img_size
        task_res[filename]['query_results'] = dict()

        for instruction in list(task_samples[filename].keys()):
            data_type = data_types[filename][instruction]
            save_path = os.path.join(args.save_dir, f"{task}/{data_type}/{filename[:-4]}")
            vis_query_text = '_'.join(instruction.replace('/', ' ').split(' '))
            cur_json_save_path = os.path.join(save_path, f'result_{vis_query_text}.json')
            if os.path.exists(cur_json_save_path):
                continue

            question = query_prompt.format(query_text=instruction, interested_key=interested_key)
            msgs = [{'role': 'user', 'content': question}]
            response, attention_outputs = eval_main.model.chat(image, msgs, eval_main.tokenizer, query_text_attention=True, sampling=False, temperature=0., num_beams=1, output_attentions=True, return_dict_in_generate=True, output_attentions_generation=True)

            gen_query_idx = find_element_content_indices(response, eval_main.tokenizer, interested_key=interested_key)
            attention_outputs['gen_query_idx']=gen_query_idx
            attention_outputs['gt_bboxes'] = query_gts[instruction]
            attention_outputs["original_image"] = image

            tag_output = eval_main.forward(attention_outputs, save_path)
            
            tag_output['data_type'] = data_type
            tag_output['prompt'] = question
            tag_output["response"] = response

            tag_res_from_prompt_acc = tag_output['tag_res_from_prompt']["final_decision"]['click_point_acc']
            click_point_acc = tag_res_from_prompt_acc
            if len(tag_output['tag_res_from_gen'])>1:
                click_point_acc = tag_output['tag_res_from_gen']["final_decision"]['click_point_acc']
                
            num_action += 1
            if click_point_acc:
                corr_action += 1
                if data_type == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
            else:
                if data_type == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
            up2now_text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
            up2now_icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
            print(f"Up2Now Acc: acc@action: {corr_action / num_action:.4f}\tacc@text: {up2now_text_acc}\tacc@icon: {up2now_icon_acc}\tnum_action:{num_action}", flush=True)
            task_res[filename]['query_results'][instruction] = tag_output
    task_res['overall'] = {
        'action_acc': corr_action / num_action,
        'total_num': num_action,
        'text_acc': sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0,
        'icon_acc': sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    }
    with open(os.path.join(args.save_dir, dataset), "w") as f:
        json.dump(task_res, f, indent=4, ensure_ascii=False)
    
    print(f"{task}: Action Acc: " + str(corr_action / num_action), flush=True)
    print(f"{task}: Total num: " + str(num_action), flush=True)
    print(f"{task}: Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0), flush=True)
    print(f"{task}: Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0), flush=True)