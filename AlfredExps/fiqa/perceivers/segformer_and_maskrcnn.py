import argparse
from typing import Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T

from fiqa.perceivers.basics_and_dummies import SegModelBase
from fiqa.perceivers.maskrcnn.maskrcnn import MaskRCNN
from fiqa.perceivers.segformer.segformer import SegFormer

class SegformerMaskRCNN(SegModelBase):
    """An ensemble of the two models.

    Using the validation results on our dataset, we've chosen the objects for 
    which SegFormer is better. 
    The hypothesis is that it's also better at inference on those objects.
    """

    def __init__(self, args) -> None:
        super().__init__()
        
        self.segformer = SegFormer(args)
        self.maskrcnn = MaskRCNN(args)
        self.maskrcnn_aux_transform = T.Compose([T.ToTensor()])

        # 'Fridge' was specially deleted 
        # since SegFormer badly detects an opened fridge
        self.segformer_better_on = {
            'AlarmClock', 'ArmChair', 'BathtubBasin', 'Bed', 'Book', 'Bowl', 
            'Box', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cloth', 
            'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 
            'Desk', 'DeskLamp', 'DiningTable', 'Drawer', 'Dresser', 'Egg', 
            'FloorLamp',  # 'Fridge' 
            'GarbageCan', 'HandTowel', 'Kettle', 
            'Knife', 'Ladle', 'Laptop', 'LightSwitch', 'Mug', 'Newspaper', 
            'Ottoman', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 
            'Pillow', 'Plate', 'RemoteControl', 'Safe', 'SaltShaker', 
            'ScrubBrush', 'Shelf', 'ShowerDoor', 'SideTable', 'SinkBasin', 
            'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'StoveKnob', 'TVStand', 
            'TennisRacket', 'Toilet', 'ToiletPaper', 'Towel', 'Watch' 
        }
        self.last_choice = None

    def forward(self, rgb: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError()

    def get_interaction_mask(
        self, rgb: Union[torch.Tensor, np.ndarray], class_name: str
    ) -> np.ndarray:
        if class_name in self.segformer_better_on:
            self.last_choice = self.segformer
            return self.segformer.get_interaction_mask(rgb, class_name)
        else:
            self.last_choice = self.maskrcnn
            rgb = self.maskrcnn_aux_transform(rgb)
            return self.maskrcnn.get_interaction_mask(rgb, class_name)

    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        self.last_choice.visualize_seg_results(rgb, steps_taken, show_mask)


# Here we use the fact that SegFormer doesn't need "outside" transforms
# but MaskRCNN does, so we added `self.maskrcnn_aux_transform`
def build(args: argparse.Namespace) -> Tuple[SegModelBase, T.Compose]:
    return SegformerMaskRCNN(args), T.Compose([])
