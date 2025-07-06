import argparse
import os
import numpy as np
from PIL import Image
import torch

from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper
from core_utils.tag import AttentionDrivenGrounding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='output_single')
    parser.add_argument('--vis-flag', action='store_true', help='visualize merged attention map')
    parser.add_argument('--tokenwise-vis', action='store_true', help='visualize each token attention map')
    parser.add_argument('--filter_thresold', type=float, default=0.5)
    parser.add_argument('--topk_layer_head', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    model = MiniCPMV.from_pretrained(args.mllm_path, torch_dtype=torch.float16)
    tokenizer = PreTrainedTokenizerFastWrapper.from_pretrained(args.mllm_path)
    model = model.to('cuda')
    model.eval()

    tag = AttentionDrivenGrounding(
        tokenizer,
        topk_layer_head=args.topk_layer_head,
        filter_thresold=args.filter_thresold,
        vis_flag=args.vis_flag or args.tokenwise_vis,
    )
    tag.save_path = args.save_dir

    image = Image.open(args.image).convert('RGB')
    question_prompt = 'What is the bounding box of "'
    msgs = [{'role': 'user', 'content': question_prompt}]
    model_inputs, slice_images, best_grid, high_res_image = model.get_vllm_embeddings_for_tag(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
    )

    query_ids = torch.from_numpy(np.array(tokenizer.encode(args.query), dtype=np.int32))
    outputs = model.get_self_attentions(
        model_inputs,
        [query_ids],
        slice_images,
        best_grid,
        high_res_image,
    )
    outputs['original_image'] = image

    res = tag.forward(**outputs)

    if args.tokenwise_vis and 'tokens' in res:
        token_dir = os.path.join(args.save_dir, 'tokenwise')
        tag.plot_tokenwise_attn_maps(
            res['attn_map_list_low'],
            res['attn_map_list_high'],
            res['tokens'],
            [res['low_res_image'], res['high_res_image']],
            token_dir,
        )

    print(res['final_bbox'])


if __name__ == '__main__':
    main()
