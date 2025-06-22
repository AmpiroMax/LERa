import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np

from fiqa.task_handlers.subtask_manager import build as build_subtask_manager
from fiqa.language_processing.subtask import Subtask
from fiqa.language_processing import subtasks_helper
from fiqa.task_handlers.interactor import InteractorBase


class Planner:
    """This is the planner from the article."""

    def __init__(
            self, args: argparse.Namespace, interactor: Optional[InteractorBase] = None
        ) -> None:
        self.subtask_manager = build_subtask_manager(args, interactor)
        self.planner_type = args.planner
        if self.planner_type == 'no_replan':
            # Load processed instructions from the file
            self.instructions_processed = subtasks_helper.load_processed_instructions(
                args.split, args.instr_type, args.dataset,
                args.path_to_instructions, args.use_gt_instrs
            )

    def get_subtask_queue(self, task_info) -> List[Subtask]:
        if self.planner_type == 'no_replan':
            return self.instructions_processed[task_info]
        else:
            return self.subtask_manager.get_subtask_queue(task_info)

    def reset(self, subtask_queue: List[Subtask]):
        self.subtask_manager.reset(subtask_queue)

    def get_cur_subtask(self) -> Subtask:
        return self.subtask_manager.get_cur_subtask()
    
    def get_help_for_nav(self, rgb: np.ndarray, subtask: Subtask) -> str:
        return self.subtask_manager.get_help_for_nav(rgb, subtask)

    def replan(self, rgb: np.ndarray, err: str) -> Tuple[Subtask, Dict]:
        info = self.subtask_manager.replan_subtasks(rgb, err)
        return self.subtask_manager.get_cur_subtask(), info

    def plan_for_nav(self, checker_verdict: bool) -> Tuple[Subtask, bool, bool]:
        """Plans navigational subtasks.

        It returns the last navigational subtask in the case of failed interaction. So, 
        if the current subtask is navigational and it fails, then the current subtask 
        is returned.

        Parameters
        ----------
        checker_verdict : bool
            Verdict of the used subtask checker.

        Returns
        -------
        subtask : Subtask
            Subtask to execute.
        task_finished : bool
            True if the global task was finished.
        bool
            True if it is required to retry navigation.
        """
        if checker_verdict:
            subtask, task_finished = self.subtask_manager.get_next_subtask()
        else:
            subtask = self.subtask_manager.get_last_nav_subtask()
            task_finished = False
        return subtask, task_finished, not checker_verdict
    
    def plan_for_interaction(self, checker_verdict: bool) -> Tuple[Subtask, bool, bool]:
        """Plans interaction subtasks.

        If the current interaction subtask fails, then the current subtask is returned.
        Otherwise, the next subtask is requested.

        Parameters
        ----------
        checker_verdict : bool
            Verdict of the used subtask checker.

        Returns
        -------
        subtask : Subtask
            Subtask to execute.
        task_finished : bool
            True if the global task was finished.
        bool
            True if it is required to retry navigation.
        """
        if checker_verdict:
            subtask, task_finished = self.subtask_manager.get_next_subtask()
        else:
            subtask = self.subtask_manager.get_cur_subtask()
            task_finished = False
        return subtask, task_finished, not checker_verdict
