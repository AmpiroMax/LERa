from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
import torchvision.transforms as T
import numpy as np

from utils.logger import logger


# =============================== Segmentation ===============================
class SegModelBase(torch.nn.Module, metaclass=ABCMeta):
    """Base class for segmentation models."""

    def __init__(self) -> None:
        super().__init__()
        self.seg_results = None
        self.mask = None

    @abstractmethod
    def forward(self, rgb: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError()

    @staticmethod
    def _check_zero_mask(mask_size: bool, class_name: str) -> bool:
        if not mask_size:
            msg = f'Zero mask! seg_model has not found {class_name}'
            logger.log_warning(msg)

    @abstractmethod
    def get_interaction_mask(
        self, rgb: Union[torch.Tensor, np.ndarray], class_name: str,
        use_area_as_score: bool = False, check_zero_mask: bool = True
    ) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def ignore_objects_on_recep(self, obj_class, recep_class) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_semantic_seg(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        raise NotImplementedError()


class DummySegModel(SegModelBase):
    """A dummy model for debugging purposes."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, rgb: np.ndarray) -> None:
        self.seg_results = rgb

    def get_interaction_mask(
        self, rgb: np.ndarray, class_name: str, use_area_as_score: bool = False,
        check_zero_mask: bool = True
    ) -> np.ndarray:
        self.mask = np.ones((rgb.shape[1], rgb.shape[2]), dtype=bool)
        return self.mask

    def ignore_objects_on_recep(self, obj_class: str, recep_class: str) -> None:
        pass  # Can't ignore

    def get_semantic_seg(self) -> np.array:
        raise NotImplementedError()  # Can't implement without sensible segmentation

    def visualize_seg_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        pass  # Visualization is useless in that case


def build_dummy_seg_model() -> Tuple[torch.nn.Module, T.Compose]:
    return DummySegModel(), T.Compose([T.ToTensor()])


# ============================= Depth estimation =============================
class DepthModelBase(torch.nn.Module, metaclass=ABCMeta):
    """Base class for depth models."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class DummyDepthModel(DepthModelBase):
    """A dummy model for debugging purposes."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb


def build_dummy_depth_model() -> Tuple[torch.nn.Module, T.Compose]:
    return DummyDepthModel(), T.Compose([T.ToTensor()])
