import argparse
from typing import Tuple, List

import torch
import torchvision.transforms as T
import numpy as np
import skimage.morphology

import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor

from fiqa.perceivers.basics_and_dummies import SegModelBase
from utils.logger import logger


class SegFormer(SegModelBase):
    """
    SegFormer model trained for ALFRED scenes.
    More info about SegFormer: https://github.com/NVlabs/SegFormer
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.device = torch.device(
            f'cuda:{args.interactor_gpu}' if torch.cuda.is_available() else 'cpu'
        )

        config = mmcv.Config.fromfile(
            'fiqa/perceivers/segformer/SegFormer/'
            + 'local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py'
        )
        config.model.decode_head.num_classes = 93
        config.model.pretrained = None
        config.model.train_cfg = None
        config.data.train.dataset.pipeline[1]['reduce_zero_label'] = False
        config.data.train.dataset.pipeline[7]['seg_pad_val'] = 0
        self.model = build_segmentor(
            config.model, test_cfg=config.model.train_cfg
        )
        checkpoint = load_checkpoint(
            self.model,
            'fiqa/checkpoints/segformer_iter_96000.pth',
            map_location='cpu'
        )
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.model.PALETTE = checkpoint['meta']['PALETTE']

        self.model.cfg = config  # save the config in the model for convenience
        self.model.to(self.device)
        self.model.eval()
        self.class2id = {
            class_name: i for i, class_name in enumerate(self.model.CLASSES)
        }

    def forward(self, rgb: np.ndarray) -> List[np.ndarray]:
        return inference_segmentor(self.model, rgb)

    def get_interaction_mask(
        self, rgb: np.ndarray, class_name: str
    ) -> np.ndarray:
        self.seg_results = self(rgb)
        target_obj_id = self.get_class_id(class_name)
        self.mask = (self.seg_results[0] == target_obj_id)
        self.mask = skimage.morphology.binary_dilation(
            self.mask, skimage.morphology.disk(4)
        )
        SegFormer._check_zero_mask(np.any(self.mask), class_name)
        return self.mask

    # We especially don't use `.get()` here, since all the classes to be
    # present. And if a class is not present, the LLM have mistaken
    def get_class_id(self, class_name: str) -> int:
        if class_name not in self.class2id:
            msg = 'SegFormer does not have class {class_name}!'
            logger.log_error(msg)
            assert False, msg
        return self.class2id[class_name]

    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        logger.save_img(
            self.model.show_result(rgb, self.seg_results),
            img_name=f'sem_seg_{str(steps_taken)}'
        )
        if show_mask:
            logger.save_img(
                self.mask * 255, img_name=f'mask_{str(steps_taken)}'
            )


def build(args: argparse.Namespace) -> Tuple[SegModelBase, T.Compose]:
    return SegFormer(args), T.Compose([])
