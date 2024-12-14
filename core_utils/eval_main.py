import os
import numpy as np

import torch
from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper
from core_utils.tag import AttentionDrivenGrounding


class EvalMain:
    def __init__(self, 
                 mllm_path, 
                 tag_param_dict,
                 query_prompt, 
                 batchsize=10, 
                 vis_flag=False):
        self.load_mllm(mllm_path)
        self.query_prompt = query_prompt
        self.tag_param_dict = tag_param_dict
        print(f"TAG param_dict: {tag_param_dict}")
        self.tag = AttentionDrivenGrounding(self.tokenizer, **tag_param_dict)
        self.batchsize = batchsize
        self.vis_flag = vis_flag

    def load_mllm(self, mllm_path):
        model = MiniCPMV.from_pretrained(mllm_path, torch_dtype=torch.float16)
        tokenizer = PreTrainedTokenizerFastWrapper.from_pretrained(mllm_path)
        model = model.to(device='cuda')
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        print(f"MLLM is loaded from {mllm_path}!")

    def forward_batch(self, query_candidates, image, save_path):
        if isinstance(query_candidates, dict):
            query_text_candidates = list(query_candidates.keys())
        else:
            raise LookupError(f'query_candidiates should be in dict type')

        # tokenize all query-text
        query_text_id_list = []
        for query_text in query_text_candidates:
            query_text_id_list.append(torch.from_numpy(np.array(self.tokenizer.encode(query_text), dtype=np.int32)))
        
        # feed forward visual encoder
        question=self.query_prompt[:self.query_prompt.find('"')+1]
        msgs = [{'role': 'user', 'content': question}]
        model_inputs, slice_images, best_grid, high_res_image = self.model.get_vllm_embeddings_for_tag(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
        )

        tag_output = dict()
        tag_output['input_image_size'] = image.size
        tag_output['high_res_image_size'] = high_res_image.size
        tag_output['query_results'] = dict()

        click_point_correctness_upbnd_list = []
        prev_inputs_embeds = model_inputs["inputs_embeds"]
        prev_input_ids = model_inputs['input_ids']
        self.tag.save_path = save_path
        for start_idx in range(0, len(query_text_id_list), self.batchsize):
            model_inputs["inputs_embeds"] = prev_inputs_embeds
            model_inputs['input_ids'] = prev_input_ids
            outputs = self.model.get_self_attentions(
                model_inputs,
                query_text_id_list[start_idx:start_idx+self.batchsize],
                slice_images,
                best_grid,
                high_res_image,
            )
            outputs["original_image"] = image
            self_attentions = outputs['self_attentions']
            input_ids = outputs['input_ids']
            interested_query_idx = outputs['interested_query_idx']

            for _query_text, _self_attention, _input_ids, _interested_query_idx in zip(query_text_candidates[start_idx:start_idx+self.batchsize], self_attentions, input_ids, interested_query_idx):
                potential_gt_bboxes = query_candidates[_query_text]

                tag_output['query_results'][_query_text] = dict()
                outputs['self_attentions'] = _self_attention
                outputs['input_ids'] = _input_ids
                outputs['interested_query_idx'] = _interested_query_idx
                outputs['gt_bboxes'] = potential_gt_bboxes
                ground_res = self.tag.forward(**outputs)
                pred_bboxes = ground_res['final_bbox']

                tag_output['query_results'][_query_text]['gt_bbox'] = potential_gt_bboxes
                tag_output['query_results'][_query_text]['pred_bbox'] = pred_bboxes
                tag_output['query_results'][_query_text]['pred_bbox_coverage_rate'] = []
                tag_output['query_results'][_query_text]['pred_click_point'] = []
                tag_output['query_results'][_query_text]['pred_click_point_correctness'] = []
                tag_output['query_results'][_query_text]['pred_click_area_correctness'] = []

                pred_click_point_correctness_upbnd = False
                pred_click_area_correctness_upbnd = False

                if len(pred_bboxes) > 0:
                    for _pred_bbox, _ in pred_bboxes:
                        coverage_rate_max = self.get_max_coverage_rate(potential_gt_bboxes, _pred_bbox)
                        tag_output['query_results'][_query_text]['pred_bbox_coverage_rate'].append(coverage_rate_max)
                        click_area_in_gt = True if coverage_rate_max > 0 else False
                        
                        click_point = [(_pred_bbox[0]+_pred_bbox[2])/2, (_pred_bbox[1]+_pred_bbox[3])/2]
                        click_point_in_gt = False
                        for _gt_bbox in potential_gt_bboxes:
                            if _gt_bbox[0] <= click_point[0] <= _gt_bbox[2] and _gt_bbox[1] <= click_point[1] <= _gt_bbox[3]:
                                click_point_in_gt = True

                        tag_output['query_results'][_query_text]['pred_click_point'].append(click_point)
                        tag_output['query_results'][_query_text]['pred_click_point_correctness'].append(click_point_in_gt)
                        tag_output['query_results'][_query_text]['pred_click_area_correctness'].append(click_area_in_gt)
                        
                        pred_click_point_correctness_upbnd = click_point_in_gt or pred_click_point_correctness_upbnd
                        pred_click_area_correctness_upbnd = click_area_in_gt or pred_click_area_correctness_upbnd
                tag_output['query_results'][_query_text]['pred_click_point_correctness_upbnd'] = pred_click_point_correctness_upbnd
                tag_output['query_results'][_query_text]['pred_click_area_correctness_upbnd'] = pred_click_area_correctness_upbnd
                click_point_correctness_upbnd_list.append(pred_click_point_correctness_upbnd)

                # get final decision
                tag_output['query_results'][_query_text]["final_decision"] = dict()
                if len(pred_bboxes) <= 1:
                    tag_output['query_results'][_query_text]["final_decision"]['click_point'] = tag_output['query_results'][_query_text]['pred_click_point']
                    tag_output['query_results'][_query_text]["final_decision"]['click_point_acc'] = pred_click_point_correctness_upbnd
                    tag_output['query_results'][_query_text]["final_decision"]['click_area_acc'] = pred_click_area_correctness_upbnd
                else:
                    pred_click_area_correctness = tag_output['query_results'][_query_text]["pred_click_area_correctness"]
                    pred_click_point_correctness = tag_output['query_results'][_query_text]["pred_click_point_correctness"]
                    pred_click_point = tag_output['query_results'][_query_text]["pred_click_point"]
                    
                    # select among the most confident and then biggest bbox
                    calc_area = lambda bbox: (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    pred_bbox_area_confid = np.array([[calc_area(bbox_info[0]), bbox_info[1]] for bbox_info in pred_bboxes])
                    sorted_indices = sorted(range(len(pred_bbox_area_confid)), key=lambda i: (pred_bbox_area_confid[i][1], pred_bbox_area_confid[i][0]), reverse=True)
                    top_idx = sorted_indices[0]
                    click_point_acc_top = pred_click_point_correctness[top_idx]
                    click_area_acc_top = pred_click_area_correctness[top_idx]
                    click_point_top = pred_click_point[top_idx]
                    tag_output['query_results'][_query_text]["final_decision"]['click_point'] = click_point_top
                    tag_output['query_results'][_query_text]["final_decision"]['click_point_acc'] = click_point_acc_top
                    tag_output['query_results'][_query_text]["final_decision"]['click_area_acc'] = click_area_acc_top

        tag_output['query_results_final'] = dict()
        tag_output['query_results_final']['mean_acc_click_point_upbnd'] = np.mean(click_point_correctness_upbnd_list)
        return tag_output

    def forward(self, attention_outputs, save_path):
        cur_query_res = {
            "gt_bbox": attention_outputs["gt_bboxes"],
            "tag_res_from_prompt": dict(),
            "tag_res_from_gen": dict()
        }

        self_attentions = attention_outputs['self_attentions']
        generated_text = attention_outputs.pop('generated_text')
        gen_query_idx = attention_outputs.pop('gen_query_idx')
        
        attention_outputs['self_attentions'] = [self_attentions[0]]
        self.tag.save_path = os.path.join(save_path, 'query_text')
        tag_res_from_prompt = self.tag.forward(**attention_outputs)
        self.get_prediction(tag_res_from_prompt, attention_outputs, cur_query_res['tag_res_from_prompt'])
        
        if len(gen_query_idx)>0:
            attention_outputs['self_attentions'] = self_attentions[1:]
            attention_outputs['interested_query_idx'] = gen_query_idx
            attention_outputs['input_ids'] = [self.tokenizer.encode(generated_text)]
            attention_outputs['use_gen_attn'] = True
            try:
                assert len(attention_outputs['self_attentions']) == len(attention_outputs['input_ids'][0]), ">"*20+f"attn-len:{len(attention_outputs['self_attentions'])}, id-len: {len(attention_outputs['input_ids'][0])}"+"<"*20
            except:
                # tokenizer.decode([25765]) -> '!"\n'
                # tokenizer.decode([0,702]) -> '!"\n'
                pass
            self.tag.save_path = os.path.join(save_path, 'gen_text')
            tag_res_from_gen = self.tag.forward(**attention_outputs)
            self.get_prediction(tag_res_from_gen, attention_outputs, cur_query_res['tag_res_from_gen'])
        return cur_query_res
    
    def get_max_coverage_rate(self, potential_gt_bboxes, pred_bbox):
        coverage_rate_max = -1
        for _gt_bbox in potential_gt_bboxes:
            coverage_rate = self.compute_coverage_rate(_gt_bbox, pred_bbox)
            if coverage_rate > coverage_rate_max:
                coverage_rate_max = coverage_rate
        return coverage_rate_max
    
    @staticmethod
    def compute_coverage_rate(box1, box2):
        """The coverage ratio of bounding box 1 by bounding box 2.
        bbox = [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Calculate the coordinates of the intersection region
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        # Check if there is any overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # Calculate the area of the intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # Calculate the area of box1
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        # Calculate the coverage ratio
        coverage = intersection_area / box1_area
        return coverage

    def get_prediction(self, tag_res, attention_outputs, output_dict):
        potential_gt_bboxes = attention_outputs["gt_bboxes"]
        pred_bboxes = tag_res['final_bbox']

        output_dict['pred_bbox'] = pred_bboxes
        output_dict['pred_bbox_coverage_rate'] = []
        output_dict['pred_click_point'] = []
        output_dict['pred_click_point_correctness'] = []
        output_dict['pred_click_area_correctness'] = []

        pred_click_point_correctness_upbnd = False
        pred_click_area_correctness_upbnd = False

        if len(pred_bboxes) > 0:
            for _pred_bbox, _ in pred_bboxes:
                coverage_rate_max = self.get_max_coverage_rate(potential_gt_bboxes, _pred_bbox)
                output_dict['pred_bbox_coverage_rate'].append(coverage_rate_max)
                click_area_in_gt = True if coverage_rate_max > 0 else False

                click_point = [(_pred_bbox[0]+_pred_bbox[2])/2, (_pred_bbox[1]+_pred_bbox[3])/2]
                click_point_in_gt = False
                for _gt_bbox in potential_gt_bboxes:
                    if _gt_bbox[0] <= click_point[0] <= _gt_bbox[2] and _gt_bbox[1] <= click_point[1] <= _gt_bbox[3]:
                        click_point_in_gt = True

                output_dict['pred_click_point'].append(click_point)
                output_dict['pred_click_point_correctness'].append(click_point_in_gt)
                output_dict['pred_click_area_correctness'].append(click_area_in_gt)
                
                pred_click_point_correctness_upbnd = click_point_in_gt or pred_click_point_correctness_upbnd
                pred_click_area_correctness_upbnd = click_area_in_gt or pred_click_area_correctness_upbnd
        output_dict['pred_click_point_correctness_upbnd'] = pred_click_point_correctness_upbnd
        output_dict['pred_click_area_correctness_upbnd'] = pred_click_area_correctness_upbnd
        
        # get final decision
        output_dict["final_decision"] = dict()
        if len(pred_bboxes) <= 1:
            output_dict["final_decision"]['click_point'] = output_dict['pred_click_point']
            output_dict["final_decision"]['click_point_acc'] = pred_click_point_correctness_upbnd
            output_dict["final_decision"]['click_area_acc'] = pred_click_area_correctness_upbnd
        else:
            pred_click_area_correctness = output_dict["pred_click_area_correctness"]
            pred_click_point_correctness = output_dict["pred_click_point_correctness"]
            pred_click_point = output_dict["pred_click_point"]
            
            # select among the most confident and then biggest bbox
            calc_area = lambda bbox: (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            pred_bbox_area_confid = np.array([[calc_area(bbox_info[0]), bbox_info[1]] for bbox_info in pred_bboxes])
            sorted_indices = sorted(range(len(pred_bbox_area_confid)), key=lambda i: (pred_bbox_area_confid[i][1], pred_bbox_area_confid[i][0]), reverse=True)
            top_idx = sorted_indices[0]
            click_point_acc_top = pred_click_point_correctness[top_idx]
            click_area_acc_top = pred_click_area_correctness[top_idx]
            click_point_top = pred_click_point[top_idx]
            output_dict["final_decision"]['click_point'] = click_point_top
            output_dict["final_decision"]['click_point_acc'] = click_point_acc_top
            output_dict["final_decision"]['click_area_acc'] = click_area_acc_top