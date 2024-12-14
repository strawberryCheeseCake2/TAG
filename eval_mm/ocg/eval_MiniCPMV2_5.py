import os
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import glob
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper


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
    parser.add_argument("--sampling", action='store_true', default=False, help="do sampling for mllm")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--image_aspects", type=parse_image_aspects, help="a list of aspect ratios in the format [[w1, h1], [w2, h2], ...]")
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")

    args = parser.parse_args()
    return args

args = parse_args()

model = MiniCPMV.from_pretrained(args.mllm_path, torch_dtype=torch.float16)
tokenizer = PreTrainedTokenizerFastWrapper.from_pretrained(args.mllm_path)
model = model.to(device='cuda')
model.eval()


def rescale_bbox(bbox, img_size):
    img_w, img_h = img_size
    resize_ratio_w = img_w / 1000
    resize_ratio_h = img_h / 1000
    return {
        'x1': int(bbox['x1']*resize_ratio_w),
        'y1': int(bbox['y1']*resize_ratio_h),
        'x2': int(bbox['x2']*resize_ratio_w),
        'y2': int(bbox['y2']*resize_ratio_h)
    }

def extract_bbox(text, img_size):
    res_bbox = []
    res_click_point = []

    pattern = r"(?:<box>|\[)(\d+) (\d+) (\d+) (\d+)(?:</box>|\])"
    matches = re.findall(pattern, text)
    if len(matches)>0:
        for match in matches:
            pred_bbox = rescale_bbox({
                'x1': max(int(match[0]), 0),
                'y1': max(int(match[1]), 0),
                'x2': max(int(match[2]), 0),
                'y2': max(int(match[3]), 0)
            }, img_size)
            click_point = [(pred_bbox['x2']+pred_bbox['x1'])//2, (pred_bbox['y2']+pred_bbox['y1'])//2]
            res_bbox.append(pred_bbox)
            res_click_point.append(click_point)
    return res_bbox, res_click_point

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection
    return intersection / union

def is_in_gt(gt_bbox, click_point):
    if gt_bbox[0] <= click_point[0] <= gt_bbox[2] and gt_bbox[1] <= click_point[1] <= gt_bbox[3]:
        return True
    else:
        return False

def plot_bbox(input_image, pred_res, img_path, gt_res):
    image_for_check = input_image.copy()
    draw = ImageDraw.Draw(image_for_check)
    font= ImageFont.truetype(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NotoSansCJK-Black.ttc"), 10)
    for _pred_res in pred_res:
        pred_bbox, click_point, query_text, iou = _pred_res
        if pred_bbox == {}:
            continue
        bbox_x1 = pred_bbox['x1']
        bbox_y1 = pred_bbox['y1']
        bbox_x2 = pred_bbox['x2']
        bbox_y2 = pred_bbox['y2']
        draw.rectangle([(bbox_x1, bbox_y1), (bbox_x2, bbox_y2)], outline ="red")
        draw.text((bbox_x1, bbox_y1), f"{query_text}_{iou:.2f}", fill="red", font=font)
        if click_point:
            x, y = click_point
            radius = 5
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill="red")
    
    for _gt_res in gt_res:
        _gt_bbox, query_text =_gt_res
        draw.rectangle([(_gt_bbox[0], _gt_bbox[1]), (_gt_bbox[2], _gt_bbox[3])], outline ="blue")
        draw.text((_gt_bbox[0], _gt_bbox[1]), query_text, fill="blue", font=font)
    image_for_check.save(img_path)


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

query_prompt = """What is the bounding box of '{query_text}' in the image? The bounding box output format is: <box>xmin ymin xmax ymax</box>. Please directly output the bounding box."""
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
    gt_bboxes = []
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
        gt_bboxes.append([_ocr_loc, query_text])
    
    one_example["minicpm"] = dict()
    one_example["minicpm"]['query_results'] = dict()

    pred_res = []
    iou_list = []
    click_point_acc = []
    for query_text in list(ocr_list_new.keys()):
        one_example["minicpm"]['query_results'][query_text] = dict()
        potential_gt_bboxes = ocr_list_new[query_text]
        one_example["minicpm"]['query_results'][query_text]['gt_bbox'] = potential_gt_bboxes

        question=query_prompt.format(query_text=query_text)
        msgs = [{'role': 'user', 'content': question}]
        res = model.chat(image=crop_image, msgs=msgs, tokenizer=tokenizer, sampling=args.sampling)
        one_example["minicpm"]['query_results'][query_text]['raw_output'] = res
        pred_bboxes, click_points = extract_bbox(res, img_size=(in_img_w, in_img_h))
        one_example["minicpm"]['query_results'][query_text]['pred_bboxes'] = pred_bboxes
        one_example["minicpm"]['query_results'][query_text]['pred_bboxes_iou'] = []
        one_example["minicpm"]['query_results'][query_text]['pred_click_point'] = click_points
        one_example["minicpm"]['query_results'][query_text]['pred_click_point_acc'] = []

        if len(pred_bboxes) > 0:
            iou_max = -1
            click_acc = False
            _pred_res = []
            for pred_bbox, click_point in zip(pred_bboxes, click_points):
                pred_loc = [pred_bbox["x1"], pred_bbox["y1"], pred_bbox["x2"], pred_bbox["y2"]]
                _iou_max = -1
                _click_acc = False
                for _gt_bbox in potential_gt_bboxes:
                    iou = compute_iou(_gt_bbox, pred_loc)
                    acc = is_in_gt(_gt_bbox, click_point)
                    if iou > _iou_max:
                        _iou_max = iou
                    if acc:
                        _click_acc = acc
                one_example["minicpm"]['query_results'][query_text]['pred_bboxes_iou'].append(_iou_max)
                one_example["minicpm"]['query_results'][query_text]['pred_click_point_acc'].append(_click_acc)
                if _iou_max > iou_max:
                    iou_max = _iou_max
                    _pred_res = [pred_bbox, click_point, query_text, _iou_max]
                if _click_acc:
                    click_acc = _click_acc
            pred_res.append(_pred_res)
        else:
            iou_max = 0
            click_acc = False
        iou_list.append(iou_max)
        click_point_acc.append(click_acc)
    one_example["minicpm"]["pred_iou_upbound"] = iou_list
    one_example["minicpm"]["pred_iou_upbound_mean"] = np.mean(iou_list)
    one_example["minicpm"]["pred_click_point_acc_upbound"] = np.mean(click_point_acc)
    global_click_point_correctness_list.extend(click_point_acc)
    if args.vis_flag:
        plot_bbox(crop_image, pred_res, os.path.join(save_path, website_name+".jpg"), gt_bboxes)

    with open(json_save_path, "w") as f:
        json.dump(one_example, f, indent=4, ensure_ascii=False)


data_dir = args.data_path
path_list = glob.glob(os.path.join(data_dir, "*", "*.json"))
path_list.sort()
image_aspects = args.image_aspects
for image_aspect in image_aspects:
    global_click_point_correctness_list = []
    for idx, json_path in tqdm(enumerate(path_list), desc='MiniCPMV2.5-eval', total=len(path_list)):
        session_id = json_path.split("/")[-2]
        with open(json_path, 'r') as f:
            one_example = json.load(f)
        one_example["saved_image_path"] = os.path.join(data_dir, session_id, "images", one_example["saved_image_name"])
        bbox_pred(one_example, json_path, image_aspect=image_aspect)
    print(f"image_aspect: {image_aspect},\t ACC@click-point:{np.mean(global_click_point_correctness_list)}", flush=True)



