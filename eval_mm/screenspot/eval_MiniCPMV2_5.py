import os
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import copy
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument("--sampling", action='store_true', default=False, help="do sampling for mllm")
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
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

def transform_gt_bbox(original_img_size, new_img_size, gt_bbox):
    w_ratio = new_img_size[0]/original_img_size[0]
    h_ratio = new_img_size[1]/original_img_size[1]
    gt_bbox[0] *= w_ratio
    gt_bbox[1] *= h_ratio
    gt_bbox[2] *= w_ratio
    gt_bbox[3] *= h_ratio
    return gt_bbox

def plot_bbox(input_image, pred_res, gt_res, img_path):
    image_for_check = input_image.copy()
    draw = ImageDraw.Draw(image_for_check)
    font= ImageFont.truetype(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NotoSansCJK-Black.ttc"), 10)
    pred_bbox, click_point, query_text, iou = pred_res
    if pred_bbox:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = pred_bbox
        draw.rectangle([(bbox_x1, bbox_y1), (bbox_x2, bbox_y2)], outline ="red")
        draw.text((bbox_x1, bbox_y1), f"{query_text}_{iou:.2f}", fill="red", font=font)
    if click_point:
        x, y = click_point
        radius = 5
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill="red")
    
    _gt_bbox, query_text = gt_res
    draw.rectangle([(_gt_bbox[0], _gt_bbox[1]), (_gt_bbox[2], _gt_bbox[3])], outline ="blue")
    draw.text((_gt_bbox[0], _gt_bbox[1]), query_text, fill="blue", font=font)
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))
    image_for_check.save(img_path)


question_template="""In this UI screenshot, I want to perform the command "{task_prompt}", which element should I perform? Please output in json format ONLY with the following three keys: action_type, element_content_or_description, box. If the element is an icon and there is no text content on this element, please output up to three keywords for the element_content_or_description. The box value format is: <box>xmin ymin xmax ymax</box>."""


tasks = ["mobile", "desktop", "web"]
for task in tasks:
    task_res = dict()
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)), flush=True)
    
    task_res = list()
    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    for j, item in tqdm(enumerate(screenspot_data)):
        item_res = dict()
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found", flush=True)
            input()
        original_image = Image.open(img_path).convert("RGB")
        original_img_size = original_image.size
        image = resize_image(original_image, original_img_size)
        in_img_w, in_img_h = image.size
        instruction = item["instruction"]
        original_bbox = item["bbox"]
        original_bbox = [original_bbox[0], original_bbox[1], original_bbox[0] + original_bbox[2], original_bbox[1] + original_bbox[3]]
        bbox = transform_gt_bbox(original_img_size, (in_img_w, in_img_h), copy.deepcopy(original_bbox))
        item_res['original_bbox'] = copy.copy(original_bbox)
        item_res['bbox'] = copy.copy(bbox)

        question = question_template.format(task_prompt=instruction)
        msgs = [{'role': 'user', 'content': question}]
        response = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, sampling=args.sampling)
        pred_bboxes, click_points = extract_bbox(response, img_size=(in_img_w, in_img_h))
        item_res['pred_bbox'] = pred_bboxes
        cur_item_correct = 0
        if len(click_points) > 0:
            click_point = click_points[0] # only evaluate the first bbox when multiple bboxes are predicted

            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                print("match " + str(corr_action / num_action), flush=True)
                cur_item_correct = 1
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                print("unmatch " + str(corr_action / num_action), flush=True)
        else:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            print("Step: " + str(j) + " wrong format")
            click_point = [-1, -1]

        up2now_text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        up2now_icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
        print(f"Up2Now Acc: acc@action: {corr_action / num_action:.4f}\tacc@text: {up2now_text_acc}\tacc@icon: {up2now_icon_acc}\tnum_action:{num_action}", flush=True)
        
        item_res['filename'] = filename
        item_res['data_type'] = item["data_type"]
        item_res['data_source'] = item["data_source"]
        item_res['instruction'] = instruction
        item_res['original_img_size'] = original_img_size
        item_res['img_size'] = [in_img_w, in_img_h]
        item_res['response'] = response
        item_res['correct'] = cur_item_correct
        item_res['click_point'] = click_point
        task_res.append(item_res)

        if args.vis_flag:
            save_path = f"{task}/{filename[:-4]}_{'_'.join(instruction.split(' '))}.png"
            plot_bbox(image, [None, click_point, instruction, -1], [item_res['bbox'], instruction], os.path.join(args.save_dir, save_path))

    with open(os.path.join(args.save_dir, dataset), "w") as f:
        json.dump(task_res, f, indent=4, ensure_ascii=False)

    print(task + ": Action Acc: " + str(corr_action / num_action))
    print(task + ": Total num: " + str(num_action))
    print(task + ": Wrong format num: " + str(num_wrong_format))
    print(task + ": Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    print(task + ": Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))