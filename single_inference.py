import argparse
from PIL import Image

from core_utils.eval_main import EvalMain
from eval_mm.screenspot.prompt_tag import query_prompt, interested_key, find_element_content_indices


def resize_image(img, max_size=1344):
    W, H = img.size
    if W <= max_size and H <= max_size:
        return img
    if W >= H:
        W_new = max_size
        H_new = int(H / W * W_new)
    else:
        H_new = max_size
        W_new = int(W / H * H_new)
    return img.resize((W_new, H_new), Image.Resampling.BICUBIC)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TAG grounding on a single screenshot")
    parser.add_argument('--mllm_path', type=str, required=True, help='Path to MiniCPM-Llama3-V2.5 model')
    parser.add_argument('--image', type=str, required=True, help='Path to screenshot image')
    parser.add_argument('--query', type=str, required=True, help='Command text to ground')
    parser.add_argument('--save-dir', type=str, default='output_single', help='Directory to store optional outputs')
    parser.add_argument('--vis-flag', action='store_true', help='Visualize intermediate results')
    parser.add_argument('--filter-threshold', type=float, default=0.5)
    parser.add_argument('--topk-layer-head', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    tag_params = {
        "filter_thresold": args.filter_threshold,
        "topk_layer_head": args.topk_layer_head,
        "vis_flag": args.vis_flag,
    }

    eval_main = EvalMain(
        mllm_path=args.mllm_path,
        tag_param_dict=tag_params,
        query_prompt=query_prompt,
        vis_flag=args.vis_flag,
    )

    image = Image.open(args.image).convert("RGB")
    image = resize_image(image)

    question = query_prompt.format(query_text=args.query, interested_key=interested_key)
    msgs = [{'role': 'user', 'content': question}]

    response, attention_outputs = eval_main.model.chat(
        image,
        msgs,
        eval_main.tokenizer,
        query_text_attention=True,
        sampling=False,
        temperature=0.0,
        num_beams=1,
        output_attentions=True,
        return_dict_in_generate=True,
        output_attentions_generation=True,
    )

    gen_query_idx = find_element_content_indices(response, eval_main.tokenizer, interested_key=interested_key)
    attention_outputs['gen_query_idx'] = gen_query_idx
    attention_outputs['gt_bboxes'] = []
    attention_outputs['original_image'] = image

    tag_output = eval_main.forward(attention_outputs, args.save_dir)

    if len(tag_output['tag_res_from_gen']) > 1:
        final_decision = tag_output['tag_res_from_gen']['final_decision']
    else:
        final_decision = tag_output['tag_res_from_prompt']['final_decision']

    print("Model Response:", response)
    print("Predicted click point:", final_decision['click_point'])
    print("Predicted bounding boxes:", tag_output['tag_res_from_prompt'].get('pred_bbox'))


if __name__ == '__main__':
    main()
