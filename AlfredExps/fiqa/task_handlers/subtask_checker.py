from abc import ABCMeta, abstractmethod
import argparse
from typing import Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
import skimage.morphology

from fiqa.alfred_thor_env import AlfredThorEnv
from fiqa.language_processing.subtask import Subtask
from fiqa.task_handlers.interactor import InteractorBase
from fiqa.perceivers.basics_and_dummies import SegModelBase

from fiqa.task_handlers.vqa_models.model_zoo import prepare_model, do_forward_pass
from fiqa.language_processing.subtasks_helper import generate_questions_from_subtask

from utils.logger import logger


class SubtaskCheckerBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.prev_steps_taken = -1

    @abstractmethod
    def reset(self) -> None:
        """Resets the checker. Used before every episode."""
        raise NotImplementedError()

    @staticmethod
    def log(subtask: Subtask, steps_taken: int, verdict: bool) -> bool:
        """Logs the necessary info using `utils.logger.logger`.

        N.B. Should be used at the end of `check()` or in the `return` statement.
        """
        logger.log({
            'time': datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f'),
            'subtask': subtask,
            'action': subtask.action,
            'steps_taken': steps_taken,
            'success': f'Checker:{verdict}',
            'error': ''
        })
        return verdict

    @abstractmethod
    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        """Checks whether the subtask was successfully completed using 
        the last RGB-image.

        The variable `steps_taken` is used to detect the infinite loop: when 
        the executor does nothing and just considers the subtask 
        to be completed.

        Parameters
        ----------
        rgb : np.ndarray
            The last RGB-image obtained from the AI2THOR simulator.
        subtask : Subtask
            A subtask to check (navigation or one of 7 interactions).
        steps_taken : int
            A number of steps taken up to that moment.

        Returns
        -------
        verdict : bool
            Whether the task was successfully completed in the opinion 
            of the checker.
        """
        raise NotImplementedError()


class DummySubtaskChecker(SubtaskCheckerBase):
    """This `SubtaskChecker` always returns 'True', so it doesn't actually check 
    anything."""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        pass

    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        return self.log(subtask, steps_taken, verdict=True)


class OracleSubtaskChecker(SubtaskCheckerBase):
    """This `SubtaskChecker` uses GT info from the AI2THOR simulator."""

    def __init__(self, args: argparse.Namespace, env: AlfredThorEnv) -> None:
        super().__init__()
        self.args = args
        self.env = env

    def reset(self) -> None:
        self.prev_steps_taken = -1

    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        if self.prev_steps_taken == steps_taken:
            logger.log_warning('Infinite loop detected!')
            return self.log(subtask, steps_taken, verdict=True)
        self.prev_steps_taken = steps_taken

        if (
            self.args.existence_only_checker 
            and subtask.action != 'GotoLocation'
        ):
            return self.log(subtask, steps_taken, verdict=True)  # No check

        if subtask.action == 'GotoLocation':
            for obj in self.env.last_event.metadata['objects']:
                if obj['visible'] and obj['objectType'] == subtask.obj:
                    return self.log(subtask, steps_taken, verdict=True)
            return self.log(subtask, steps_taken, verdict=False)
        elif subtask.action == 'ToggleObjectOn' and subtask.obj == 'Faucet':
            # Since ALFRED performs additional actions to clean objects, 
            # we can't use metadata['lastActionSuccess']
            verdict = any(
                obj['isToggled'] for obj in self.env.last_event.metadata['objects'] 
                if obj['objectType'] == 'Faucet'
            )
            return self.log(subtask, steps_taken, verdict)
        elif self.env.last_event.metadata['lastAction'] != subtask.action:
            # This means no action was performed in AI2THOR and
            # ThorEnv.va_interact() returned False.
            # Note that the condition doesn't captures cases where a subtask
            # was tried to execute several times. But in those cases we have to
            # re-execute it anyway, so having
            # self.env.last_event.metadata['lastActionSuccess'] == False
            # doesn't contradict this.
            return self.log(subtask, steps_taken, verdict=False)
        return self.log(
            subtask, steps_taken, 
            verdict=self.env.last_event.metadata['lastActionSuccess']
        )


class OracleSubtaskCheckerWithNoise(OracleSubtaskChecker):
    """This `SubtaskChecker` makes mistake with a given probability."""

    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        if self.prev_steps_taken == steps_taken:
            logger.log_warning('Infinite loop detected!')
            return self.log(subtask, steps_taken, verdict=True)
        self.prev_steps_taken = steps_taken

        if (
            self.args.existence_only_checker 
            and subtask.action != 'GotoLocation'
        ):
            return self.log(subtask, steps_taken, verdict=True)  # No check

        verdict = None
        if subtask.action == 'GotoLocation':
            if self.args.interaction_only_checker:
                return self.log(subtask, steps_taken, verdict=True)  # No check
            else:
                for obj in self.env.last_event.metadata['objects']:
                    if obj['visible'] and obj['objectType'] == subtask.obj:
                        verdict = True
                        break
                verdict = False
        elif subtask.action == 'ToggleObjectOn' and subtask.obj == 'Faucet':
            # Since ALFRED performs additional actions to clean objects, 
            # we can't use metadata['lastActionSuccess']
            verdict = any(
                obj['isToggled'] for obj in self.env.last_event.metadata['objects'] 
                if obj['objectType'] == 'Faucet'
            )
        elif self.env.last_event.metadata['lastAction'] != subtask.action:
            # This means no action was performed in AI2THOR and
            # ThorEnv.va_interact() returned False.
            # Note that the condition doesn't captures cases where a subtask
            # was tried to execute several times. But in those cases we have to
            # re-execute it anyway, so having
            # self.env.last_event.metadata['lastActionSuccess'] == False
            # doesn't contradict this.
            verdict = False
        else:
            verdict = self.env.last_event.metadata['lastActionSuccess']

        # Flip the verdict to model mistakes
        if np.random.uniform() > self.args.checker_correctness_prob:
            verdict = not verdict

        return self.log(
            subtask, steps_taken, verdict=verdict
        )


class VQASubtaskChecker(SubtaskCheckerBase):
    """`VQAChecker` uses a VQA model to ask specific questions about 
    an RGB-image in order to check the subtask."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.obj_in_hands = None
        self.device = torch.device(
            f'cuda:{args.vqa_gpu}' if torch.cuda.is_available() else 'cpu'
        )

        self.vqa_model, self.img_transform_vqa = prepare_model(args)
        self.vqa_model = self.vqa_model.to(self.device).eval()
        self.vqa_model_name = args.vqa_model

    def reset(self) -> None:
        self.prev_steps_taken = -1
        self.obj_in_hands = None

    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        # Infinite loop defence
        if self.prev_steps_taken == steps_taken:
            logger.log_warning('Infinite loop detected!')
            return self.log(subtask, steps_taken, verdict=True)
        self.prev_steps_taken = steps_taken

        if (
            self.args.existence_only_checker
            and subtask.action != 'GotoLocation'
        ):
            return self.log(subtask, steps_taken, verdict=True)  # No check

        img = self.img_transform_vqa(rgb).unsqueeze(0).to(self.device)
        question = generate_questions_from_subtask(
            subtask, self.obj_in_hands
        )
        verdict = do_forward_pass(
            self.vqa_model, self.vqa_model_name, img, question
        )
        if verdict and subtask.action == 'PickupObject':
            self.obj_in_hands = subtask.obj
        if verdict and subtask.action == 'PutObject':
            self.obj_in_hands = None

        return self.log(subtask, steps_taken, verdict)


class SegAndVQASubtaskChecker(VQASubtaskChecker):
    """This checker is the `VQASubtaskChecker` enhanced with 
    a segmentation model that is used to check the navigation. 
    
    Such enhancement is better in terms of consistency since segmentation 
    is used to obtain an interaction mask right after a navigational subtask.
    """

    def __init__(
        self, args: argparse.Namespace, 
        seg_tuple: Tuple[SegModelBase, T.Compose]
    ) -> None:
        super().__init__(args)
        self.seg_model, self.img_transform_seg = seg_tuple

    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        # Infinite loop defence
        if self.prev_steps_taken == steps_taken:
            logger.log_warning('Infinite loop detected!')
            return self.log(subtask, steps_taken, verdict=True)
        self.prev_steps_taken = steps_taken

        if (
            self.args.existence_only_checker
            and subtask.action != 'GotoLocation'
        ):
            return self.log(subtask, steps_taken, verdict=True)  # No check

        # Here the segmentation model is used to check the navigation and 
        # the VQA model to check the interaction
        if subtask.action == 'GotoLocation':
            seg_rgb = self.img_transform_seg(rgb)
            mask = self.seg_model.get_interaction_mask(seg_rgb, subtask.obj)
            verdict = np.any(mask)
        else:
            img = self.img_transform_vqa(rgb).unsqueeze(0).to(self.device)
            question = generate_questions_from_subtask(
                subtask, self.obj_in_hands
            )

            verdict = do_forward_pass(
                self.vqa_model, self.vqa_model_name, img, question
            )
            if verdict and subtask.action == 'PickupObject':
                self.obj_in_hands = subtask.obj
            if verdict and subtask.action == 'PutObject':
                self.obj_in_hands = None

        return self.log(subtask, steps_taken, verdict)


class FramesDiffBasedSubtaskChecker(SubtaskCheckerBase):
    """This checker uses the difference of two frames to determine 
    whether a subtask execution succeeded.

    N.B. Using this checker for navigation is pointless.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.open_close_threshold = 500  # is taken from FILM
        self.default_threshold = 100  # is taken from FILM
        self.prev_rgb = None

    def reset(self) -> None:
        self.prev_rgb = None
    
    def check(
        self, rgb: np.ndarray, subtask: Subtask, steps_taken: int
    ) -> bool:
        if self.args.existence_only_checker:
            assert False, (
                'FramesDiffBasedSubtaskChecker is useless with option'
                + 'args.existence_only_checker is set to "True"!'
            )

        if subtask.action == 'GotoLocation':
            self.prev_rgb = rgb
            return self.log(subtask, steps_taken, verdict=True)  # No check

        wheres = np.where(self.prev_rgb != rgb)
        wheres_ar = np.zeros(self.prev_rgb.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions) + 1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)

        threshold = \
            self.open_close_threshold \
            if subtask.action in ['OpenObject', 'CloseObject'] \
            else self.default_threshold
        verdict = max_area > threshold
        self.prev_rgb = rgb
        return self.log(subtask, steps_taken, verdict)


def build(
    args: argparse.Namespace, 
    interactor: Optional[InteractorBase] = None, 
    env: Optional[AlfredThorEnv] = None
) -> SubtaskCheckerBase:
    if args.checker == 'none':
        return DummySubtaskChecker()
    elif args.checker == 'oracle':
        assert 'tests' not in args.split, \
            'Oracle checker is not allowed on test splits!'
        assert env is not None, 'Oracle navigator needs env but it is None!'
        return OracleSubtaskChecker(args, env)
    elif args.checker == 'oracle_with_noise':
        assert 'tests' not in args.split, \
            'Oracle checker is not allowed on test splits!'
        assert env is not None, 'Oracle navigator needs env but it is None!'
        return OracleSubtaskCheckerWithNoise(args, env)
    elif args.checker == 'vqa':
        return VQASubtaskChecker(args)
    elif args.checker == 'seg_and_vqa':
        assert interactor is not None, 'An interactor must be provided!'
        # Currently we have only segmentation based interactors and they have
        # `seg_model` and `img_transform_seg` fields
        seg_tuple = (interactor.seg_model, interactor.img_transform_seg)
        return SegAndVQASubtaskChecker(args, seg_tuple)
    elif args.checker == 'frames_diff_based':
        assert not args.existence_only_checker, \
            'This checker can not check navigation. Please, use another.'
        return FramesDiffBasedSubtaskChecker(args)
    else:
        assert False, f'Unknown checker {args.checker}'
