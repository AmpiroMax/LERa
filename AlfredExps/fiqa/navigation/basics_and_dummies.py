from abc import ABCMeta, abstractmethod
import argparse
from typing import Union, List, Optional

import numpy as np
import torch

from fiqa.language_processing.subtask import Subtask


class NavigatorBase(metaclass=ABCMeta):
    actions_list = [
        'RotateLeft', 'RotateRight', 'MoveAhead',
        'LookUp', 'LookDown', 'StopNav'
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abstractmethod
    def reset(self, subtask_queue: Optional[List[Subtask]] = None) -> None:
        """Resets the navigator. Used before every episode."""
        raise NotImplementedError()

    @abstractmethod
    def reset_before_new_objective(
        self, subtask: Subtask, retry_nav: bool
    ) -> None:
        """Used to reset parts of the navigator to search for a new object.

        Parameters
        ----------
        subtask : Subtask
            A navigational subtask to execute.
        retry_nav : bool
            True if it is required to retry navigation to the goal object.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, rgb: Union[torch.Tensor, np.ndarray]
    ) -> Union[str, dict]:
        """Predicts the next navigational action.

        Parameters
        ----------
        rgb : torch.Tensor or np.ndarray
            A processed or raw RGB to use.

        Returns
        -------
        action : str or dict
            A predicted action. Normally navigators predict a string, but 
            `OracleNavigator` uses teleportation, so it returns a dict.
        """
        raise NotImplementedError()


class RandomNavigator(NavigatorBase):
    """A simple navigator that predicts navigation actions randomly."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

    def reset(self, subtask_queue: Optional[List[Subtask]] = None) -> None:
        pass

    def reset_before_new_objective(
        self, subtask: Subtask, retry_nav: bool
    ) -> None:
        pass

    def __call__(self, rgb: np.ndarray) -> str:
        return np.random.choice(
            NavigatorBase.actions_list,
            p=[0.19, 0.19, 0.19, 0.19, 0.19, 0.05]
        )
