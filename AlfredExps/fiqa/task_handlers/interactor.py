from abc import ABCMeta, abstractmethod
import argparse
from typing import Tuple, Optional

from collections import deque
import numpy as np
import torchvision.transforms as T
import skimage.morphology

from fiqa.alfred_thor_env import AlfredThorEnv
from fiqa.language_processing.subtask import Subtask
from fiqa.perceivers.model_zoo import build_seg_model
from alfred_utils.gen.constants import SCREEN_WIDTH, SCREEN_HEIGHT


class InteractorBase(metaclass=ABCMeta):
    """Base class for all interactors."""

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_interaction_mask(
        self, rgb: np.ndarray, subtask: Subtask, retry_interaction: Optional[bool] = None
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def visualize_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        raise NotImplementedError()


class SegBasedInteractor(InteractorBase):
    """Base class for all interactors that use segmentation."""

    def __init__(
        self, args: argparse.Namespace, env: Optional[AlfredThorEnv] = None
    ) -> None:
        super().__init__()
        self.seg_model, self.img_transform_seg = build_seg_model(args, env)

    def reset(self) -> None:
        pass
    
    def visualize_results(
        self, rgb: np.ndarray, steps_taken: int, show_mask: bool = True
    ) -> None:
        self.seg_model.visualize_seg_results(rgb, steps_taken, show_mask)


class TrivialSegBasedInteractor(SegBasedInteractor):
    """This interactor implements trivial logic and doesn't store any 
    additional info. 
    An interaction mask is obtained each time from segmentation.
    """

    def __init__(
        self, args: argparse.Namespace, env: Optional[AlfredThorEnv] = None
    ) -> None:
        super().__init__(args, env)

    def get_interaction_mask(
        self, rgb: np.ndarray, subtask: Subtask, retry_interaction: Optional[bool] = None
    ) -> np.ndarray:
        seg_rgb = self.img_transform_seg(rgb)
        return self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)


class AdvancedSegBasedInteractor(SegBasedInteractor):
    """This interactor uses the same hacks as in FILM. 
    
    Main hacks for interaction are 1) the mask used for opening and turning on 
    is used for closing and turning off, 2) the mask obtained from 
    the difference of two consecutive frames in the case of 
    successful 'PutObject' is used for picking up if 'PickupObject' fails. 
    Suprisingly, but it is not guaranteed that the opened/turned on object 
    obtains the biggest mask.

    N.B. This implementation assumes that closing/turning off happens after
    opening/turning on (although it can handle cases like in 449th ep. when 
    the agent is asked to close a 'Laptop' without opening it).
    """

    def __init__(
        self, args: argparse.Namespace, env: Optional[AlfredThorEnv] = None
    ) -> None:
        super().__init__(args, env)
        self.opening_or_toggling_mask = None
        self.prev_subtask = None
        self.hack_was_used = False
        self.prev_rgb = None
        self.pickup_mask_from_put = None  # <==> FILM's self.put_rgb_mask

    def reset(self) -> None:
        self.opening_or_toggling_mask = None
        self.prev_subtask = None
        self.hack_was_used = False
        self.prev_rgb = None
        self.pickup_mask_from_put = None

    def get_interaction_mask(
        self, rgb: np.ndarray, subtask: Subtask, retry_interaction: bool
    ) -> np.ndarray:
        if retry_interaction:
            if subtask.action == 'PickupObject' and self.pickup_mask_from_put is not None:
                mask = self.seg_model.mask = self.pickup_mask_from_put
                self.pickup_mask_from_put = None
            elif self.hack_was_used:
                seg_rgb = self.img_transform_seg(rgb)
                mask = self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)
                self.hack_was_used = False
            elif self.seg_model.mask.sum() < 1e-3:  # Retry because of zero mask
                # Make mistake anyway, so to avoid the infinite loop:
                # StopNav --> Zero mask --> None mask --> StopNav --> ...
                seg_rgb = self.img_transform_seg(rgb)
                mask = self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)
                self.opening_or_toggling_mask = None
                self.pickup_mask_from_put = None
            else:
                mask = None  # Refuse to retry interaction
                # Since a sensible navigator changes the agent's pos and view
                # when re-executing a navigational subtask,
                # the opening or toggling mask becomes invalid
                self.opening_or_toggling_mask = None
                self.pickup_mask_from_put = None
        else:
            self.hack_was_used = False
            if (
                self.opening_or_toggling_mask is not None
                and (
                    subtask.action in ['CloseObject', 'ToggleObjectOff']
                    or (
                        subtask.action == 'PutObject'
                        and self.prev_subtask.action == 'OpenObject'
                    )
                )
            ):
                mask = self.seg_model.mask = self.opening_or_toggling_mask
                self.hack_was_used = True
                if subtask.action != 'PutObject':
                    self.opening_or_toggling_mask = None
            else:
                seg_rgb = self.img_transform_seg(rgb)
                mask = self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)
                if subtask.action in ['OpenObject', 'ToggleObjectOn']:
                    self.opening_or_toggling_mask = mask

            if (
                self.prev_subtask is not None
                and self.prev_subtask.action == 'PutObject'
            ):
                self.pickup_mask_from_put = \
                    AdvancedSegBasedInteractor.mask_from_two_frames_diff(
                        self.prev_rgb, rgb
                    )
        self.prev_subtask = subtask
        self.prev_rgb = rgb
        return mask
    
    # Is taken from https://github.com/soyeonm/FILM/blob/08c9afade79d0b0995d034dc6cf770cf8d37f070/models/segmentation/segmentation_helper.py#L462
    @staticmethod
    def mask_from_two_frames_diff(
        f1: np.ndarray, f2: np.ndarray
    ) -> np.ndarray:
        diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
        diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])
        diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
        diff_mask = np.zeros((300, 300))
        diff_mask[diff1] = 1.0
        diff_mask[diff2] = 1.0
        diff_mask[diff3] = 1.0

        # Divide diff mask into 2 regions with scikit image
        connected_regions = skimage.morphology.label(diff_mask, connectivity=2)
        unique_labels = [i for i in range(0, np.max(connected_regions) + 1)]
        lab_area = {lab: 0 for lab in unique_labels}
        min_ar = 10000000
        smallest_lab = None
        for lab in unique_labels:
            wheres = np.where(connected_regions == lab)
            lab_area[lab] = len(wheres[0])
            if lab_area[lab] > 100:
                min_ar = min(lab_area[lab], min_ar)
                if min_ar == lab_area[lab]:
                    smallest_lab = lab

        return_mask = np.zeros((300, 300))
        if not(smallest_lab) is None:
            return_mask[np.where(connected_regions == smallest_lab)] = 1
        return return_mask


class OracleSegBasedInteractor(SegBasedInteractor):
    """This interactor is the oracle version of `AdvancedSegBasedInteractor`.
    It always closes and turns off the object that was opened/turned on.

    N.B. This implementation assumes that closing/turning off happens after
    opening/turning on (although it can handle cases like in 449th ep. when 
    the agent is asked to close a 'Laptop' without opening it and 
    **the previous subtask.action wasn't CloseObject**).
    """

    def __init__(self, args: argparse.Namespace, env: AlfredThorEnv) -> None:
        super().__init__(args, env)
        self.env = env
        self.openings_and_toggles = deque()
        print(
            "Warning! "
            + "This interactor hasn't been tested in case it has to retry "
            + "the last action!"
        )

    def reset(self) -> None:
        self.openings_and_toggles = deque()

    def get_interaction_mask(
        self, rgb: np.ndarray, subtask: Subtask, retry_interaction: Optional[bool] = None
    ) -> np.ndarray:
        if (
            len(self.openings_and_toggles) > 0
            and (
                subtask.action in ['CloseObject', 'ToggleObjectOff']
                or (
                    subtask.action == 'PutObject'
                    and subtask.obj == self.openings_and_toggles[-1][0]
                    and not self.openings_and_toggles[-1][1]
                )
            )
        ):
            # Determine which object was opened/toggled on
            key = 'isToggled' if subtask.action == 'ToggleObjectOff' \
                else 'isOpen'
            target_obj = None
            for obj in self.env.last_event.metadata['objects']:
                if (
                    obj[key]
                    and obj['objectType'] == self.openings_and_toggles[-1][0]
                ):
                    target_obj = obj['objectId']
                    break
            if (
                target_obj is not None
                and target_obj not in self.env.last_event.instance_masks
            ):
                mask = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=bool)
            else:
                mask = self.env.last_event.instance_masks[target_obj]
            self.seg_model.mask = mask  # For correct visualization of the mask
            # The interactor hopes the action will succeed
            if subtask.action != 'PutObject':
                self.openings_and_toggles[-1][1] = True
        else:
            seg_rgb = self.img_transform_seg(rgb)
            mask = self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)
            # The last element must be removed if the last
            # put/close/toggle action is considered to be successful
            if (
                len(self.openings_and_toggles) > 0
                and self.openings_and_toggles[-1][1]
            ):
                self.openings_and_toggles.pop()
            if subtask.action in ['OpenObject', 'ToggleObjectOn']:
                self.openings_and_toggles.append([subtask.obj, False])
        return mask


def build_interactor(
    args: argparse.Namespace, env: Optional[AlfredThorEnv] = None
) -> Tuple[InteractorBase, T.Compose]:
    if args.interactor == 'trivial_seg_based':
        return TrivialSegBasedInteractor(args, env)
    elif args.interactor == 'advanced_seg_based':
        return AdvancedSegBasedInteractor(args, env)
    elif args.interactor == 'oracle_seg_based':
        assert 'tests' not in args.split, 'Oracle interactor is not allowed on test splits!'
        assert args.seg_model == 'oracle', \
            'OracleSegBasedInteractor only works with oracle segmentation!'
        assert env is not None, \
            'Env must be set when using OracleSegBasedInteractor!'
        return OracleSegBasedInteractor(args, env)
    else:
        assert False, f'Unknown interactor name: {args.interactor}'
