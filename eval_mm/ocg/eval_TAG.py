import os
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import glob
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from PIL import Image

from core_utils.eval_main import EvalMain


def parse_image_aspects(aspect_string):
    try:
        aspects = json.loads(aspect_string)
        if all(isinstance(lst, list) and len(lst) == 2 for lst in aspects):
            return aspects
        raise ValueError
    except:
        raise argparse.ArgumentTypeError("Invalid format for image aspects. Expected a list of [width, height] pairs.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--image_aspects", type=parse_image_aspects, help="a list of aspect ratios in the format [[w1, h1], [w2, h2], ...]")
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")
    # params for TAG
    parser.add_argument("--filter_thresold", type=float, default=0.5)
    parser.add_argument("--topk_layer_head", type=int, default=10, help="select topk image-attn-weight from all layer heads")
    parser.add_argument("--batchsize", type=int, default=10)

    args = parser.parse_args()
    return args

args = parse_args()

query_prompt = """What is the bounding box of "{query_text}" in the image?"""
tag_params = {
        "filter_thresold": args.filter_thresold,
        "topk_layer_head": args.topk_layer_head,
        "vis_flag": args.vis_flag,
    }
eval_main = EvalMain(mllm_path=args.mllm_path, tag_param_dict=tag_params, query_prompt=query_prompt, batchsize=args.batchsize, vis_flag=args.vis_flag)


IGNORED_OCR = [
    "<", ">", ".", "V", "&", "%", "$", "#", "!", "~", "`", "^", "&", "*", "(", ")", "-", "_", "=", "+", "{", "}", "[", "]", "\\", "|", "'", "\"", ":", ";", "?", "/", ",", "."
]

def is_meaningless(s):
    """Define a function to check whether a string contains meaningless repetitions or special characters"""
    if len(s)<2:
        return True
    # Use a regular expression to check for patterns of 2 to 4 consecutive identical characters
    if len(s) < 5 and re.search(r"(.)\1{1,5}", s):
        return True
    # Count the number of special characters
    count_special = sum(1 for char in s if char in IGNORED_OCR)
    # Determine if the proportion of special characters is too high (> 80% of the total characters)
    if count_special > len(s) * 0.8:
        return True
    return False

global_click_point_correctness_list = []
def bbox_pred(one_example, json_path, image_aspect=[4, 3], crop_at=[0, 0]):
    json_path_list = json_path.split('/')
    website_name = json_path_list[-2]
    save_path = os.path.join(args.save_dir, f'{image_aspect[0]}_{image_aspect[1]}', website_name)
    json_save_path = os.path.join(save_path, json_path_list[-1])
    if os.path.exists(json_save_path):
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_path = one_example["saved_image_path"]
    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size
    if image_aspect[0] > image_aspect[1]:
        if image_w < image_h:
            real_w = image_w
            real_h = int(image_w/image_aspect[0]*image_aspect[1])
        else:
            real_w = min(int(image_h/image_aspect[1]*image_aspect[0]), image_w)
            real_h = min(int(real_w/image_aspect[0]*image_aspect[1]), image_h)
    else:
        if image_w < image_h:
            real_h = image_w
        else:
            real_h = image_h
        real_w = int(real_h/image_aspect[1]*image_aspect[0])
    crop_image = image.crop((crop_at[0], crop_at[1], real_w, real_h))
    
    ocr_str = one_example['ocr_en']
    in_img_w, in_img_h = real_w, real_h
    pattern = r"\((\d+, \d+, \d+, \d+)\): (.+)"
    ocr_list = ocr_str.split(';')
    ocr_list_new = defaultdict(list)
    for _ocr in ocr_list:
        match = re.match(pattern, _ocr.strip())
        if match:
            coordinates_str = match.group(1)
            query_text = match.group(2)
        else:
            continue
        _ocr_loc = [int(num) for num in coordinates_str.split(", ")]
        # transform mid-x, mid-y into lef-top x, left-top y
        _ocr_loc[0] = max(0, _ocr_loc[0] - _ocr_loc[2]//2)
        _ocr_loc[1] = max(0, _ocr_loc[1] - _ocr_loc[3]//2)
        _ocr_loc[2] = _ocr_loc[0] + _ocr_loc[2]
        _ocr_loc[3] = _ocr_loc[1] + _ocr_loc[3]
        if _ocr_loc[2] > in_img_w or _ocr_loc[3] > in_img_h:
            continue
        query_text = query_text.strip()
        if is_meaningless(query_text):
            continue
        ocr_list_new[query_text].append(_ocr_loc)

    outputs = eval_main.forward_batch(ocr_list_new, crop_image, save_path)
    global_click_point_correctness_list.extend([outputs['query_results'][query_text]["final_decision"]['click_point_acc'] for query_text in list(outputs['query_results'].keys())])

    one_example["TAG"] = outputs
    with open(json_save_path, "w") as f:
        json.dump(one_example, f, indent=4, ensure_ascii=False)

data_dir = args.data_path
path_list = glob.glob(os.path.join(data_dir, "*", "*.json"))
path_list.sort()
image_aspects = args.image_aspects
for image_aspect in image_aspects:
    global_click_point_correctness_list = []
    for idx, json_path in tqdm(enumerate(path_list), desc='TAG-eval', total=len(path_list)):
        session_id = json_path.split("/")[-2]
        with open(json_path, 'r') as f:
            one_example = json.load(f)
        one_example["saved_image_path"] = os.path.join(data_dir, session_id, "images", one_example["saved_image_name"])
        bbox_pred(one_example, json_path, image_aspect=image_aspect)
    print(f"image_aspect: {image_aspect},\t ACC@click-point: {np.mean(global_click_point_correctness_list)} | top{args.topk_layer_head}, lb: {args.filter_thresold}", flush=True)