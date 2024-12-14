import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import os
from PIL import Image
from collections import deque
import copy
import torch


class AttentionDrivenGrounding:
    def __init__(self, 
                 tokenizer, 
                 topk_layer_head=10,
                 filter_thresold=0.5, 
                 vis_flag=False):
        self.tokenizer = tokenizer
        self.topk_layer_head = topk_layer_head
        self.filter_thresold = filter_thresold
        self.vis_flag = vis_flag
        self.save_path = None

    def token_to_learnable_query(self, self_attentions, cross_attentions, image_bound, slice_images, best_grid, interested_query_idx):
        num_bound = len(image_bound[0])
        token_wise_topk_layer_head = dict()
        token_wise_topk_layer_head['token_wise'] = dict()
        beam_idx=0
        num_layer = len(self_attentions[0])
        num_head = len(self_attentions[0][0][beam_idx])
        TOTAL_HEADS = num_layer * num_head

        attn_map_list_low = []
        attn_map_list_high = []
        for seq_idx in interested_query_idx:
            _self_attns = self_attentions[0]

            all_heads_img_attn_weights = dict()
            for layer_idx in range(num_layer):
                for head_idx in range(num_head):
                    if self.topk_layer_head == TOTAL_HEADS:
                        all_heads_img_attn_weights[f"{layer_idx}-{head_idx}"] = [0, 0]
                        continue
                    _self_attn_per_layer = _self_attns[layer_idx][beam_idx][head_idx][seq_idx:seq_idx+1].cpu().to(torch.float32).numpy()
                    low_res_img_weight = _self_attn_per_layer[:, image_bound[0][0][0]:image_bound[0][0][1]].sum()
                    high_res_img_weight = 0
                    if num_bound > 1:
                        for img_bnd in image_bound[0][1:]:
                            high_res_img_weight += _self_attn_per_layer[:, img_bnd[0]:img_bnd[1]].sum() # 1x96
                    all_heads_img_attn_weights[f"{layer_idx}-{head_idx}"] = [low_res_img_weight, high_res_img_weight]
            if self.topk_layer_head == TOTAL_HEADS:
                topk_keys = list(all_heads_img_attn_weights.keys())
            else:
                if num_bound > 1:
                    sorted_keys = sorted(all_heads_img_attn_weights.keys(), key=lambda x: all_heads_img_attn_weights[x][1], reverse=True)
                else:
                    sorted_keys = sorted(all_heads_img_attn_weights.keys(), key=lambda x: all_heads_img_attn_weights[x][0], reverse=True)
                topk_keys = sorted_keys[:self.topk_layer_head]
            
            weight_list = []
            image_weight = []
            for layer_head_idx in topk_keys:
                layer_idx, head_idx = [int(idx) for idx in layer_head_idx.split('-')]
                _self_attn_per_layer = _self_attns[layer_idx][beam_idx][head_idx][seq_idx:seq_idx+1].cpu().to(torch.float32).numpy()
                for idx, (img_bnd, cross_attn, cur_img) in enumerate(zip(image_bound[0], cross_attentions.numpy(), slice_images)):
                    img_w, img_h = cur_img.size
                    attn_w, attn_h = img_w//14, img_h//14
                    _attn_of_img = _self_attn_per_layer[:, img_bnd[0]:img_bnd[1]].copy() # 1x96
                    if len(image_weight)<=idx:
                        image_weight.append(_attn_of_img.sum())
                    else:
                        image_weight[idx] += _attn_of_img.sum()
                    weighted_attn = _attn_of_img @ cross_attn[:, :(attn_w*attn_h)] # 1x1024
                    weighted_attn = weighted_attn.reshape(attn_h, attn_w).transpose()
                    if len(weight_list)<=idx:
                        weight_list.append(weighted_attn)
                    else:
                        weight_list[idx] += weighted_attn
            assert len(weight_list) == len(image_weight) == len(image_bound[0])
            for idx in range(len(weight_list)):
                weight_list[idx] /= self.topk_layer_head
                image_weight[idx] /= self.topk_layer_head

            attn_map_list_low.append(weight_list[0])

            if len(image_bound[0]) > 1:
                high_res_weight = self.concat_2d_arrays(weight_list[1:], best_grid)
                attn_map_list_high.append(high_res_weight)
            else:
                attn_map_list_high.append(weight_list[0])

        return attn_map_list_low, attn_map_list_high

    def token_to_learnable_query_gen(self, self_attentions, cross_attentions, image_bound, slice_images, best_grid, interested_query_idx):
        """self_attentions are from generated tokens"""
        num_bound = len(image_bound[0])
        token_wise_topk_layer_head = dict()
        token_wise_topk_layer_head['token_wise'] = dict()
        beam_idx=0
        num_layer = len(self_attentions[0])
        num_head = len(self_attentions[0][0][beam_idx])
        TOTAL_HEADS = num_layer * num_head

        attn_map_list_low = []
        attn_map_list_high = []
        for seq_idx in interested_query_idx:
            _self_attns = self_attentions[seq_idx]

            all_heads_img_attn_weights = dict()
            for layer_idx in range(num_layer):
                for head_idx in range(num_head):
                    if self.topk_layer_head == TOTAL_HEADS:
                        all_heads_img_attn_weights[f"{layer_idx}-{head_idx}"] = [0, 0]
                        continue
                    _self_attn_per_layer = _self_attns[layer_idx][beam_idx][head_idx].cpu().to(torch.float32).numpy()
                    low_res_img_weight = _self_attn_per_layer[:, image_bound[0][0][0]:image_bound[0][0][1]].sum()
                    high_res_img_weight = 0
                    if num_bound > 1:
                        for img_bnd in image_bound[0][1:]:
                            high_res_img_weight += _self_attn_per_layer[:, img_bnd[0]:img_bnd[1]].sum() # 1x96
                    all_heads_img_attn_weights[f"{layer_idx}-{head_idx}"] = [low_res_img_weight, high_res_img_weight]
            if self.topk_layer_head == TOTAL_HEADS:
                topk_keys = list(all_heads_img_attn_weights.keys())
            else:
                if num_bound > 1:
                    sorted_keys = sorted(all_heads_img_attn_weights.keys(), key=lambda x: all_heads_img_attn_weights[x][1], reverse=True)
                else:
                    sorted_keys = sorted(all_heads_img_attn_weights.keys(), key=lambda x: all_heads_img_attn_weights[x][0], reverse=True)
                topk_keys = sorted_keys[:self.topk_layer_head]
            
            weight_list = []
            image_weight = []
            for layer_head_idx in topk_keys:
                layer_idx, head_idx = [int(idx) for idx in layer_head_idx.split('-')]
                _self_attn_per_layer = _self_attns[layer_idx][beam_idx][head_idx].cpu().to(torch.float32).numpy()
                for idx, (img_bnd, cross_attn, cur_img) in enumerate(zip(image_bound[0], cross_attentions.numpy(), slice_images)):
                    img_w, img_h = cur_img.size
                    attn_w, attn_h = img_w//14, img_h//14
                    _attn_of_img = _self_attn_per_layer[:, img_bnd[0]:img_bnd[1]].copy() # 1x96
                    if len(image_weight)<=idx:
                        image_weight.append(_attn_of_img.sum())
                    else:
                        image_weight[idx] += _attn_of_img.sum()
                    weighted_attn = _attn_of_img @ cross_attn[:, :(attn_w*attn_h)] # 1x1024
                    weighted_attn = weighted_attn.reshape(attn_h, attn_w).transpose()
                    if len(weight_list)<=idx:
                        weight_list.append(weighted_attn)
                    else:
                        weight_list[idx] += weighted_attn
            assert len(weight_list) == len(image_weight) == len(image_bound[0])
            for idx in range(len(weight_list)):
                weight_list[idx] /= self.topk_layer_head
                image_weight[idx] /= self.topk_layer_head

            attn_map_list_low.append(weight_list[0])

            if len(image_bound[0]) > 1:
                high_res_weight = self.concat_2d_arrays(weight_list[1:], best_grid)
                attn_map_list_high.append(high_res_weight)
            else:
                attn_map_list_high.append(weight_list[0])

        return attn_map_list_low, attn_map_list_high
    
    def forward(self, self_attentions, cross_attentions, input_ids, image_bound, slice_images, best_grid, high_res_image, original_image, interested_query_idx, gt_bboxes=None, use_gen_attn=False):
        if use_gen_attn:
            attn_map_list_low, attn_map_list_high = self.token_to_learnable_query_gen(self_attentions, cross_attentions, image_bound, slice_images, best_grid, interested_query_idx)
        else:
            attn_map_list_low, attn_map_list_high = self.token_to_learnable_query(self_attentions, cross_attentions, image_bound, slice_images, best_grid, interested_query_idx)

        attn_map_low_all_tokens = np.stack(attn_map_list_low, 0)
        attn_map_high_all_tokens = np.stack(attn_map_list_high, 0)
        
        attn_map_low_merge = attn_map_low_all_tokens.mean(0)
        attn_map_high_merge = attn_map_high_all_tokens.mean(0)
        
        norm_max_low, norm_max_high = attn_map_low_merge.max(), attn_map_high_merge.max()
        attn_map_low_final = attn_map_low_merge / norm_max_low
        attn_map_high_final = attn_map_high_merge / norm_max_high

        attn_map_low_final = np.kron(attn_map_low_final, np.ones((14, 14)))
        attn_map_high_final = np.kron(attn_map_high_final, np.ones((14, 14)))
        assert attn_map_high_final.shape == high_res_image.size
        # Here, we only consider high-resolution parts, as including low-resolution parts will introduce some noises.
        high_res_attn_map_lb = attn_map_high_final * (attn_map_high_final >= self.filter_thresold)
        _, high_res_pred_bboxes = self.find_connected_regions_with_confidence(high_res_attn_map_lb)
        filtered_bbox_high_res = self.filter_bbox(high_res_pred_bboxes, threshold=self.filter_thresold)
        if self.vis_flag:
            interested_query_text = self.tokenizer.decode(input_ids[0][interested_query_idx[0]:interested_query_idx[-1]+1])
            vis_query_text = '_'.join(interested_query_text.replace('/', ' ').replace('"', ' ').replace(',', ' ').split(' '))
            vis_query_text = vis_query_text[:100]

            gt_bboxes_in_high_res = self.transform_bbox(high_res_image, original_image, copy.deepcopy(gt_bboxes))
            gt_bboxes_in_low_res = self.transform_bbox(slice_images[0], original_image, copy.deepcopy(gt_bboxes))
            self.plot_merged_bbox_attn_map([slice_images[0], high_res_image], [[attn_map_low_final], [attn_map_high_final]], [None, filtered_bbox_high_res], f"query_{vis_query_text[:100]}_lb{self.filter_thresold}-attnmap.jpg", [gt_bboxes_in_low_res, gt_bboxes_in_high_res])
        
        final_bbox = self.transform_bbox_with_confidence(target_image=original_image, source_image=high_res_image, pred_bboxes=copy.deepcopy(filtered_bbox_high_res))

        return {
            "low_res_image": slice_images[0],
            "high_res_image": high_res_image,
            "final_bbox": final_bbox,
            "low_res_attn_map": attn_map_low_final,
            "high_res_attn_map": attn_map_high_final,
            }
    
    @staticmethod
    def concat_2d_arrays(input_list, shape):
        w, h = shape
        if len(input_list) != w * h:
            raise ValueError("The length of the input list must be equal to w * h.")
        
        sub_shape = input_list[0].shape
        result = np.zeros((w * sub_shape[0], h * sub_shape[1]))
        for i in range(h):
            for j in range(w):
                index = i * w + j
                sub_array = np.array(input_list[index])
                result[j*sub_shape[0]:(j+1)*sub_shape[0], 
                        i*sub_shape[1]:(i+1)*sub_shape[1]] = sub_array
        
        return result
    
    def find_connected_regions_with_confidence(self, mask, cond_weight=None):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        width, height  = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        regions = []

        def bfs(start_i, start_j):
            queue = deque([(start_i, start_j)])
            region = []
            while queue:
                i, j = queue.popleft()
                if i < 0 or i >= width or j < 0 or j >= height or visited[i][j] or mask[i][j] <= 0:
                    continue
                
                visited[i][j] = True
                region.append((i, j))

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        queue.append((i + di, j + dj))
            
            return region

        # Calculate the bounding rectangle for each region
        bounding_boxes = []
        for i in range(width):
            for j in range(height):
                if mask[i][j] > 0 and not visited[i][j]:
                    region = bfs(i, j)
                    regions.append(region)

                    if region:
                        min_i = min(p[0] for p in region)
                        max_i = max(p[0] for p in region)
                        min_j = min(p[1] for p in region)
                        max_j = max(p[1] for p in region)
                        # Mark all points within the bounding box as visited
                        visited[min_i:max_i+1, min_j:max_j+1] = True
                        if cond_weight is None:
                            bbox_confid = np.max(mask[min_i:max_i+1, min_j:max_j+1])
                        else:
                            bbox_confid = np.mean(mask[min_i:max_i+1, min_j:max_j+1]) * cond_weight
                        bounding_boxes.append([[min_i, min_j, max_i, max_j], bbox_confid])

        return regions, bounding_boxes

    def filter_bbox(self, bbox_list, threshold):
        res = []
        for bbox in bbox_list:
            if bbox[1] >= threshold:
                res.append(bbox)
        return res

    def transform_bbox_with_confidence(self, target_image, source_image, pred_bboxes):
        source_w, source_h = source_image.size
        target_w, target_h = target_image.size
        w_ratio = target_w / source_w
        h_ratio = target_h / source_h
        for idx in range(len(pred_bboxes)):
            pred_bboxes[idx][0][0] *= w_ratio
            pred_bboxes[idx][0][2] *= w_ratio
            pred_bboxes[idx][0][1] *= h_ratio
            pred_bboxes[idx][0][3] *= h_ratio
        return pred_bboxes
    
    def transform_bbox(self, target_image, source_image, pred_bboxes):
        source_w, source_h = source_image.size
        target_w, target_h = target_image.size
        w_ratio = target_w / source_w
        h_ratio = target_h / source_h
        for idx in range(len(pred_bboxes)):
            pred_bboxes[idx][0] *= w_ratio
            pred_bboxes[idx][2] *= w_ratio
            pred_bboxes[idx][1] *= h_ratio
            pred_bboxes[idx][3] *= h_ratio
        return pred_bboxes

    @staticmethod
    def apply_mask_to_image(image, expanded_mask):
        heatmap = cv2.applyColorMap((expanded_mask.transpose() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        image_np = np.array(image)
        superimposed_img = cv2.addWeighted(image_np, 0.5, heatmap, 0.3, 0)
        return Image.fromarray(superimposed_img)
    
    def plot_merged_bbox_attn_map(self, image_list, bbox_attn_map_list, bbox_list, save_name, gt_bbox_list=[None, None]):
        """plot images and masks"""
        fig = plt.figure(figsize=(20, 15))
        try:
            axes = fig.subplots(2, 2)
            
            # Attention-masked low-res and high-res images
            for idx in range(2):
                if len(bbox_attn_map_list[idx]) > 0:
                    combined_attn_map = np.stack(bbox_attn_map_list[idx], 0).sum(0)
                else:
                    combined_attn_map = np.zeros(image_list[idx].size)
                masked_img = self.apply_mask_to_image(image_list[idx], combined_attn_map)
                axes[0, idx].imshow(masked_img)
                axes[0, idx].axis('off')
                axes[0, idx].set_title(f"size: {image_list[idx].size}")
                
                # Plot colormap
                im = axes[1, idx].imshow(combined_attn_map.transpose(), cmap='jet', vmin=0, vmax=1)
                axes[1, idx].axis('off')
                axes[1, idx].set_title("Attention Map")
                
                # Plot bbox
                if bbox_list[idx] is not None:
                    for bbox_info in bbox_list[idx]:
                        if len(bbox_info) == 2:
                            bbox, confid = bbox_info
                            bbox_color = 'r'
                        else:
                            bbox = bbox_info
                            bbox_color = 'blue'
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
                        axes[0, idx].add_patch(rect)
                        rect_attn = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
                        axes[1, idx].add_patch(rect_attn)
                        # Add confidence text within the bounding rectangle.
                        if len(bbox_info) == 2:
                            confid_text = f'{confid:.4f}'
                            axes[0, idx].text(x1, y2+5, confid_text, 
                                            fontsize=8, color='r', 
                                            verticalalignment='top', 
                                            horizontalalignment='left',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))
                
                if gt_bbox_list[idx] is not None:
                    for bbox_info in gt_bbox_list[idx]:
                        if len(bbox_info) == 2:
                            bbox, _ = bbox_info
                            bbox_color = 'white'
                        else:
                            bbox = bbox_info
                            bbox_color = 'white'
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
                        axes[0, idx].add_patch(rect)
                        rect_attn = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
                        axes[1, idx].add_patch(rect_attn)
                            
                # Add colorbar
                divider = make_axes_locatable(axes[1, idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            plt.subplots_adjust(hspace=0.05)
            plt.tight_layout()
            save_path = os.path.join(self.save_path, save_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        finally:
            plt.close(fig)
    