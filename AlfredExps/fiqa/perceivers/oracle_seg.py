
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T

from utils.logger import logger
import fiqa.perceivers.maskrcnn.alfworld_constants as alfworld_constants
from alfred_utils.gen.constants import SCREEN_WIDTH, SCREEN_HEIGHT

from fiqa.alfred_thor_env import AlfredThorEnv
from fiqa.perceivers.basics_and_dummies import SegModelBase


class OracleSegModel(SegModelBase):
    """Oracle segmentation model that obtains masks from 
    the AI2THOR simulator."""

    def __init__(self, env: AlfredThorEnv) -> None:
        super().__init__()
        self.env = env
        self.total_cat2idx = {
            v: k for k, v in enumerate(alfworld_constants.ALL_DETECTOR)
        }  # is updated by a navigator
        self.cat_equate_dict = dict()  # is updated by a navigator

    def forward(self, _: np.ndarray) -> None:
        self.seg_results = self.env.last_event.instance_masks

    def get_interaction_mask(
        self, rgb: np.ndarray, class_name: str, use_area_as_score: bool = True,
        check_zero_mask: bool = True
    ) -> np.ndarray:
        assert use_area_as_score, 'Using scores with oracle segmentation is meaningless.'

        self(rgb)
        mask_size = 0
        self.mask = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=bool)
        for obj_id, obj_mask in self.seg_results.items():
            if class_name in self.cat_equate_dict:
                class_name = self.cat_equate_dict
            is_target_object = False
            if 'Sliced' in class_name or class_name in ['SinkBasin', 'BathtubBasin']:
                is_target_object = class_name in obj_id
            else:
                obj_type = obj_id.split('|')[0]
                is_target_object = class_name == obj_type
            if is_target_object:
                cur_mask_size = np.sum(obj_mask)
                if cur_mask_size > mask_size:
                    mask_size = cur_mask_size
                    self.mask = obj_mask
        if check_zero_mask:
            OracleSegModel._check_zero_mask(mask_size, class_name)
        return self.mask

    def ignore_objects_on_recep(self, obj_class: str, recep_class: str) -> None:
        avoid_ids = set()
        obj_id2idx = {
            obj_metadata['objectId']: i
            for i, obj_metadata in enumerate(self.env.last_event.metadata['objects'])
        }
        for obj_id in self.seg_results.keys():
            is_target_object = False
            if 'Sliced' in obj_class or obj_class in ['SinkBasin', 'BathtubBasin']:
                is_target_object = obj_class in obj_id
            else:
                obj_type = obj_id.split('|')[0]
                is_target_object = obj_class == obj_type

            if is_target_object:
                obj_metadata = self.env.last_event.metadata['objects'][obj_id2idx[obj_id]]
                obj_receps = set(
                    recep_id.split('|')[-1] if recep_class in [
                        'SinkBasin', 'BathtubBasin'
                    ] else recep_id.split('|')[0]
                    for recep_id in obj_metadata['parentReceptacles']
                )
                if recep_class in obj_receps:
                    avoid_ids.add(obj_id)

        self.seg_results = {
            obj_id: obj_mask
            for obj_id, obj_mask in self.seg_results.items() if obj_id not in avoid_ids
        }

    def get_semantic_seg(self) -> np.array:
        semantic_seg = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, len(self.total_cat2idx)))
        for obj_id, obj_mask in self.seg_results.items():
            if 'Sliced' in obj_id or 'SinkBasin' in obj_id or 'BathtubBasin' in obj_id:
                obj_class = obj_id.split('|')[-1]
            else:
                obj_class = obj_id.split('|')[0]
            if obj_class in self.total_cat2idx:
                semantic_seg[:, :, self.total_cat2idx[obj_class]] += obj_mask
        return semantic_seg

    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        logger.save_img(
            self.env.last_event.instance_segmentation_frame,
            img_name=f'gt_sem_seg_{str(steps_taken)}'
        )
        if show_mask:
            logger.save_img(
                self.mask * 255, img_name=f'gt_mask_{str(steps_taken)}'
            )


def build_oracle_seg_model(
    env: AlfredThorEnv
) -> Tuple[torch.nn.Module, T.Compose]:
    return OracleSegModel(env), T.Compose([])
