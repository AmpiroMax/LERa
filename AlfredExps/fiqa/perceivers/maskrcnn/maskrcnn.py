"""
The code was taken from 
https://github.com/soyeonm/FILM/blob/public/models/segmentation/segmentation_helper.py 
and adapted.
"""

import argparse
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T

from alfred_utils.gen.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from fiqa.perceivers.maskrcnn.alfworld_mrcnn import load_pretrained_model
import fiqa.perceivers.maskrcnn.alfworld_constants as alfworld_constants
from fiqa.perceivers.basics_and_dummies import SegModelBase
from utils.logger import logger


# The code is specially not optimized a lot to preserve the original structure
class MaskRCNN(SegModelBase):
    """MaskRCNN that was trained for ALFWorld."""

    def __init__(self, args) -> None:
        super().__init__()
        self.device = torch.device(
            f'cuda:{args.interactor_gpu}' if torch.cuda.is_available() else 'cpu'
        )

	    # Large
        self.sem_seg_model_alfw_large = load_pretrained_model(
            'fiqa/checkpoints/alfworld_maskrcnn_receps_lr5e-3_003.pth',
            self.device, 'recep'
        )
        self.sem_seg_model_alfw_large.eval()
        self.sem_seg_model_alfw_large.to(self.device)

        self.large = alfworld_constants.STATIC_RECEPTACLES
        self.large_objects2idx = {k: i for i, k in enumerate(self.large)}
        self.large_idx2large_object = {v: k for k, v in self.large_objects2idx.items()}

	    # Small
        self.sem_seg_model_alfw_small = load_pretrained_model(
            'fiqa/checkpoints/alfworld_maskrcnn_objects_lr5e-3_005.pth',
            self.device, 'obj'
        )
        self.sem_seg_model_alfw_small.eval()
        self.sem_seg_model_alfw_small.to(self.device)

        self.small = alfworld_constants.OBJECTS_DETECTOR
        self.small_objects2idx = {k: i for i, k in enumerate(self.small)}
        self.small_idx2small_object = {v: k for k, v in self.small_objects2idx.items()}

        # Since self.agent and self.args are not accessible, 
        # we need to define the varibles actually used
        self.total_cat2idx = {
            v: k for k, v in enumerate(alfworld_constants.ALL_DETECTOR)
        }  # is updated by a navigator
        self.cat_equate_dict = dict()  # is updated by a navigator
        # self.cat_equate_dict = {
        #     'DeskLamp': 'FloorLamp', 'ButterKnife': 'Knife'
        # }
        self.sem_seg_threshold_small = 0.8  # As in FILM
        self.sem_seg_threshold_large = 0.8  # As in FILM
        self.with_mask_above_05 = True  # As in FILM
        # The code below was moved from `forward()` to `__init__()`
        self.desired_classes_small = set()
        self.desired_classes_large = set()
        for cat_name in self.total_cat2idx:
            if not(cat_name in ['None', 'fake']):
                if cat_name in self.large:
                    large_class = self.large_objects2idx[cat_name]
                    self.desired_classes_large.add(large_class)
                elif cat_name in self.small:
                    small_class = self.small_objects2idx[cat_name]
                    self.desired_classes_small.add(small_class)
        # And some additional variables
        self.palette = np.random.randint(0, 255, size=(len(self.total_cat2idx), 3))

    def forward(self, rgb: torch.Tensor) -> None:
        """Deeply based on `.get_instance_mask_seg_alfworld_both()`."""
        ims = [rgb.to(self.device)]
        results_small = self.sem_seg_model_alfw_small(ims)[0]
        results_large = self.sem_seg_model_alfw_large(ims)[0]

        inds_small = []
        inds_large = []
        for k in range(len(results_small['labels'])):
            if (
                results_small['labels'][k].item() in self.desired_classes_small
                and results_small['scores'][k] > self.sem_seg_threshold_small
            ):
                inds_small.append(k)
        for k in range(len(results_large['labels'])):
            if (
                results_large['labels'][k].item() in self.desired_classes_large
                and results_large['scores'][k] > self.sem_seg_threshold_large
            ):
                inds_large.append(k)

        pred_boxes_small = results_small['boxes'][inds_small].detach().cpu()
        pred_classes_small = results_small['labels'][inds_small].detach().cpu()
        pred_masks_small = results_small['masks'][inds_small].squeeze(1)\
            .detach().cpu().numpy()  # pred_masks[i] has shape (300,300)
        if self.with_mask_above_05:
            pred_masks_small = pred_masks_small > 0.5
        pred_scores_small = results_small['scores'][inds_small].detach().cpu()

        pred_boxes_large = results_large['boxes'][inds_large].detach().cpu()
        pred_classes_large = results_large['labels'][inds_large].detach().cpu()
        pred_masks_large = results_large['masks'][inds_large].squeeze(1)\
            .detach().cpu().numpy()  # pred_masks[i] has shape (300,300)
        if self.with_mask_above_05:
            pred_masks_large = pred_masks_large > 0.5
        pred_scores_large = results_large['scores'][inds_large].detach().cpu()

        # If a navigator asks to treat some objects as equivalent, do so
        for ci in range(len(pred_classes_small)):
            cat = self.small_idx2small_object[pred_classes_small[ci].item()]
            if cat in self.cat_equate_dict:
                pred_classes_small[ci] = self.small_objects2idx[self.cat_equate_dict[cat]]

	    # Make the above into a dictionary
        self.seg_results = self.segmented_dict = {
            'small': {
                'boxes': pred_boxes_small, 'classes': pred_classes_small,
                'masks': pred_masks_small, 'scores': pred_scores_small
            },
            'large': {
                'boxes': pred_boxes_large, 'classes': pred_classes_large,
                'masks': pred_masks_large, 'scores': pred_scores_large
            }
        }

    def get_interaction_mask(
        self, rgb: torch.Tensor, class_name: str, use_area_as_score: bool = False,
        check_zero_mask: bool = True
    ) -> np.ndarray:
        """Deeply based on `.sem_seg_get_instance_mask_from_obj_type()`."""
        # 1. Firstly, get the segmentation dict
        self(rgb)
        # 2. Secondly, get the interaction mask
        self.mask = np.zeros((300, 300))
        small_len = len(self.segmented_dict['small']['scores'])
        large_len = len(self.segmented_dict['large']['scores'])
        max_score = -1

        if class_name in self.cat_equate_dict:
            class_name = self.cat_equate_dict[class_name]
        # Get the highest score mask
        if class_name in self.large_objects2idx:
            for i in range(large_len):
                category = self.large_idx2large_object[
                    self.segmented_dict['large']['classes'][i].item()
                ]
                if category == class_name:
                    score = self.segmented_dict['large']['scores'][i]
                    if use_area_as_score:
                        score = np.sum(self.segmented_dict['large']['masks'][i])
                    max_score = max(score, max_score)
                    if max_score == score:
                        self.mask = self.segmented_dict['large']['masks'][i]
        elif class_name in self.small_objects2idx:
            for i in range(small_len):
                category = self.small_idx2small_object[
                    self.segmented_dict['small']['classes'][i].item()
                ]
                if category == class_name:
                    score = self.segmented_dict['small']['scores'][i]
                    if use_area_as_score:
                        score = np.sum(self.segmented_dict['small']['masks'][i])
                    max_score = max(score, max_score)
                    if max_score == score:
                        self.mask = self.segmented_dict['small']['masks'][i]
        else:
            msg = f'MaskRCNN does not have class {class_name}!'
            logger.log_error(msg)
            assert False, msg

        if check_zero_mask:
            MaskRCNN._check_zero_mask(np.any(self.mask), class_name)
        return self.mask

    def ignore_objects_on_recep(self, obj_class: str, recep_class: str) -> None:
        small_len = len(self.segmented_dict['small']['scores'])
        large_len = len(self.segmented_dict['large']['scores'])

        wheres = []
        for i in range(large_len):
            category = self.large_idx2large_object[
                self.segmented_dict['large']['classes'][i].item()
            ]
            if category == recep_class:
                wheres.append(self.segmented_dict['large']['boxes'][i])

        avoid_idxs = set()
        for i in range(small_len):
            category = self.small_idx2small_object[
                self.segmented_dict['small']['classes'][i].item()
            ]
            v = self.segmented_dict['small']['boxes'][i]
            if category == obj_class:
                for where in wheres:
                    if (
                        v[0] >= where[0] and v[1] >= where[1]
                        and v[2] <= where[2] and v[3] <= where[3]
                    ):
                        avoid_idxs.add(i)
                        break

        incl_idx = [i for i in range(small_len) if not(i in avoid_idxs)]
        self.segmented_dict['small']['boxes'] = \
            self.segmented_dict['small']['boxes'][incl_idx]
        self.segmented_dict['small']['classes'] = \
            self.segmented_dict['small']['classes'][incl_idx]
        self.segmented_dict['small']['masks'] = \
            self.segmented_dict['small']['masks'][incl_idx]
        self.segmented_dict['small']['scores'] = \
            self.segmented_dict['small']['scores'][incl_idx]

    def get_semantic_seg(self) -> np.array:
        small_len = len(self.segmented_dict['small']['scores'])
        large_len = len(self.segmented_dict['large']['scores'])

        semantic_seg = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, len(self.total_cat2idx)))
        for i in range(small_len):
            category = self.small_idx2small_object[
                self.segmented_dict['small']['classes'][i].item()
            ]
            v = self.segmented_dict['small']['masks'][i]
            if category in self.total_cat2idx:
                cat = self.total_cat2idx[category]
                semantic_seg[:, :, cat] +=  v.astype('float')

        for i in range(large_len):
            category = self.large_idx2large_object[
                self.segmented_dict['large']['classes'][i].item()
            ]
            v = self.segmented_dict['large']['masks'][i]
            if category in self.total_cat2idx:
                cat = self.total_cat2idx[category]
                semantic_seg[:, :, cat] += v.astype('float')

        return semantic_seg

    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        # Get (300, 300, num_classes) shape first
        cat2idx = {
            v: k for k, v in enumerate(alfworld_constants.ALL_DETECTOR)
        }
        seg_results = np.zeros((300, 300, len(cat2idx)), dtype=int)
        for i in range(len(self.seg_results['small']['scores'])):
            category = self.small_idx2small_object[
                self.seg_results['small']['classes'][i].item()
            ]
            mask = self.seg_results['small']['masks'][i]
            seg_results[:, :, cat2idx[category]] += mask.astype(int)
        for i in range(len(self.seg_results['large']['scores'])):
            category = self.large_idx2large_object[
                self.seg_results['large']['classes'][i].item()
            ]
            mask = self.seg_results['large']['masks'][i]
            seg_results[:, :, cat2idx[category]] += mask.astype(int)
        sem_seg = seg_results.argmax(axis=-1)  # (300, 300)

        # Secondly, delete the background
        for i in range(300):
            for j in range(300):
                if sem_seg[i, j] == 0:  # maybe background, maybe AlarmClock
                    if seg_results[i, j, 0] == 0:  # 100% background
                        sem_seg[i, j] = -1

        # Thirdly, color the segmentation as in SegFormer/mmseg/models/segmentors/base.py
        color_seg = np.zeros((300, 300, 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            color_seg[sem_seg == label, :] = color
        img = rgb * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        logger.save_img(img, img_name=f'sem_seg_{str(steps_taken)}')
        if show_mask:
            logger.save_img(
                self.mask * 255, img_name=f'mask_{str(steps_taken)}'
            )


def build(args: argparse.Namespace) -> Tuple[SegModelBase, T.Compose]:
    return MaskRCNN(args), T.Compose([T.ToTensor()])
