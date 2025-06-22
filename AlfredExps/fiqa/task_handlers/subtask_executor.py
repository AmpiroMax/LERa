import argparse
from typing import Tuple, List, Optional
from datetime import datetime

import numpy as np

from utils.logger import logger
from fiqa.language_processing.subtask import Subtask
from fiqa.task_handlers.navigator import build_navigator
from fiqa.task_handlers.interactor import build_interactor
from fiqa.alfred_thor_env import AlfredThorEnv


class SubtaskExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.env = AlfredThorEnv(args)

        env = None
        if args.interactor == 'oracle_seg_based' or args.seg_model == 'oracle':
            env = self.env
        self.interactor = build_interactor(args, env)
        env = None if args.navigator != 'oracle' else self.env
        (
            self.navigator,
            self.img_transform_nav,
            self.nav_retriever
        ) = build_navigator(args, env, self.interactor)

        self.args = args
        self.steps_taken = 0
        self.actseq = []
        self.fails = 0
        self.interactor_refuse_cnt = 0

    def reset(self, traj_data: dict, subtask_queue: List[Subtask]):
        """Resets the navigator and other things."""
        self.env.reset(traj_data)
        self.interactor.reset()
        if self.args.navigator == 'oracle':
            self.navigator.reset(traj_data)
        else:
            self.navigator.reset(subtask_queue)
        self.nav_retriever.reset()
        self.steps_taken = 0
        self.fails = 0
        self.actseq = []
        self.nav_success = [0, 0]
        self.reset_interactor_refuse_cnt()

    def reset_interactor_refuse_cnt(self):
        self.interactor_refuse_cnt = 0

    def update_states(self, info_for_update) -> None:
        if self.args.navigator == 'film':
            self.navigator.update_state(info_for_update)

    def _check_failures(self, success: bool, action: str):
        if not success and action != 'StopNav':
            self.fails += 1
            if self.fails >= self.args.max_fails:
                assert False, f'Interact API failed {self.args.max_fails} times'

    def _check_max_steps(self):
        if self.steps_taken >= self.args.max_steps:
            assert False, f'Maximum number of steps ({self.args.max_steps}) reached'
            
    def execute_nav_subtask(
        self, subtask: Subtask, retry_nav: bool, pose_correction_type: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """Tries to complete the given navigational subtask.

        Parameters
        ----------
        subtask : Subtask
            A subtask to execute (only navigation).
        retry_nav : bool
            True if it is required to retry navigation.
        pose_correction_type : str
            ???

        Returns
        -------
        rgb : np.array
            The latest RGB-image.
        steps_taken : int
            The number of steps taken so far.
        """
        self.nav_retriever.update_predictor_values(not retry_nav)
        stop_nav = False
        rgb = self.env.last_event.frame.copy()  # np.ndarray(uint8), (300, 300, 3)
        # Save the initial agent's view:
        if self.args.save_imgs and self.steps_taken == 0:
            img_name = str(self.steps_taken) + ("_changed_state" if self.args.change_states else "_original_state")
            logger.save_img(rgb, img_name=img_name)
        # Prepare the navigator to execute the subtask:
        if self.args.navigator == 'film':
            self.navigator.reset_before_new_objective(
                subtask, retry_nav, pose_correction_type
            )
        else:
            self.navigator.reset_before_new_objective(subtask, retry_nav)

        steps_taken_before = self.steps_taken
        while not stop_nav:
            self._check_max_steps()

            # rgb.copy() is not used since the rgb is not used further
            nav_rgb = self.img_transform_nav(rgb)
            action = self.navigator(nav_rgb)

            if action == 'StopNav' or (
                self.args.navigator == 'oracle' and action['action'] == 'StopNav'
            ):
                if steps_taken_before == self.steps_taken:  # Nav refused to execute
                    return rgb, self.steps_taken
                else:
                    success, err = None, ''
                    stop_nav = True
                    if 'tests' not in self.args.split:
                        success = self._is_goal_in_range(subtask)
                        self.nav_success[0] += success
                        self.nav_success[1] += 1
            else:
                success, _, _, err, api_action = self.env.va_interact(action)
                self.actseq.append(api_action)
                self.steps_taken += 1
                rgb = self.env.last_event.frame.copy()
                if self.args.save_imgs:
                    img_name = str(self.steps_taken) + ("_changed_state" if self.args.change_states else "_original_state")
                    logger.save_img(rgb, img_name=img_name)

            if self.args.navigator == 'oracle':
                action = action['action']
            logger.log(
                {
                    'time': datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f'),
                    'subtask': subtask,
                    'action': action,
                    'steps_taken': self.steps_taken,
                    'success':
                        f'GT:{success}' if 'tests' not in self.args.split else 'Unknown',
                    'error': err
                }
            )
            self._check_failures(success, action)
        return rgb, self.steps_taken

    def execute_interaction_subtask(
        self, subtask: Subtask, retry_interaction: bool
    ) -> Tuple[np.ndarray, int]:
        """Tries to complete the given interaction subtask.

        Parameters
        ----------
        subtask : Subtask
            A subtask to execute (one of 7 interactions).
        retry_interaction : bool
            True if it is required to retry interaction.

        Returns
        -------
        rgb : np.array
            The latest RGB-image.
        steps_taken : int
            The number of steps taken so far.
        """
        self._check_max_steps()

        rgb = self.env.last_event.frame.copy()
        mask = self.interactor.get_interaction_mask(
            rgb.copy(), subtask, retry_interaction
        )
        if mask is not None and mask.sum() < 1e-3:  # Target object was not found
            mask = self.interactor.get_interaction_mask(
                rgb.copy(), subtask, retry_interaction=True
            )

        if self.args.planner == 'with_replan':
            # It is assumed that the agent can't run into the infinite loop:
            # Nav refused -> mask is None -> Interactor refused -> Nav refused -> ...,
            # with this type of planner, so it's always allowed to refuse to execute:
            if mask is None or (mask is not None and mask.sum() < 1e-3):
                return rgb, self.steps_taken  # Refuse to execute
        else:  # "no_replan" planner
            if mask is None:  # The interactor has refused to execute the subtask
                self.interactor_refuse_cnt += 1
                if self.interactor_refuse_cnt <= 2 * self.args.max_fails:
                    return rgb, self.steps_taken  # Allow the refusal
                else:
                    # Since the agent can run into the infinite loop:
                    # Nav refused -> mask is None -> Interactor refused -> Nav refused
                    # -> ..., intentionally make an error
                    mask = np.zeros(rgb.shape[:2], dtype=bool)
            if (
                self.args.navigator == 'oracle'
                and (mask is not None and mask.sum() < 1e-3)
            ):
                # The oracle navigator guarantees to change the pose,
                # thus it is safe to refuse to execute
                return rgb, self.steps_taken

        if self.args.save_imgs:
            self.interactor.visualize_results(rgb, self.steps_taken, show_mask=True)

        success, _, _, err, api_action = self.env.va_interact(
            action=subtask.action, interact_mask=mask
        )
        self.actseq.append(api_action)
        self.steps_taken += 1
        rgb = self.env.last_event.frame.copy()
        if self.args.save_imgs:
            img_name = str(self.steps_taken) + ("_changed_state" if self.args.change_states else "_original_state")
            logger.save_img(rgb, img_name=img_name)

        logger.log(
            {
                'time': datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f'),
                'subtask': subtask,
                'action': subtask.action,
                'steps_taken': self.steps_taken,
                'success':
                    f'GT:{success}' if 'tests' not in self.args.split else 'Unknown',
                'error': err
            }
        )
        self._check_failures(success, subtask.action)
        self.nav_retriever.save_info(subtask=subtask, rgb=rgb)
        return rgb, self.steps_taken


    # Was taken from 
    # https://github.com/soyeonm/FILM/blob/public/agents/sem_exp_thor.py#L1504
    # and slightly changed
    def evaluate(self, traj_data: dict, r_idx: int, steps_taken: int):
        goal_satisfied = self.env.get_goal_satisfied()
        if goal_satisfied:
            success = True
        else:
            success = False

        pcs = self.env.get_goal_conditions_met()
        if self.fails >= self.args.max_fails and success:
            # Even if all of the conditions are met, episodes with >= 10 errors
            # are considered as not successful
            goal_satisfied = False
            success = False
            pcs = (pcs[0] - 1, pcs[1])
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(
            1.,
            path_len_weight / (float(steps_taken) if steps_taken > 0 else 1.)
        )
        pc_spl = goal_condition_success_rate * min(
            1.,
            path_len_weight / (float(steps_taken) if steps_taken > 0 else 1.)
        )

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        if not traj_data['pddl_params']['object_sliced']:
            sliced = 0
        else:
            sliced = 1

        # log success/fails
        log_entry = {
            'nav_success': self.nav_success,
            'trial': traj_data['task_id'],
            # 'scene_num': self.traj_data['scene']['scene_num'],
            'type': traj_data['task_type'],
            'repeat_idx': int(r_idx),
            'goal_instr': goal_instr,
            'completed_goal_conditions': int(pcs[0]),
            'total_goal_conditions': int(pcs[1]),
            'goal_condition_success': float(goal_condition_success_rate),
            'success_spl': float(s_spl),
            'path_len_weighted_success_spl': float(plw_s_spl),
            'goal_condition_spl': float(pc_spl),
            'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
            'path_len_weight': int(path_len_weight),
            'sliced': sliced,
            'steps_taken': steps_taken,
            # 'reward': float(reward)
        }

        return log_entry, success

    def _is_goal_in_range(self, subtask: Subtask) -> bool:
        """Checks if the target object is visible."""
        for obj in self.env.last_event.metadata['objects']:
            if obj['visible'] and obj['objectType'] == subtask.obj:
                return True
        return False

    def stop_env(self):
        self.env.stop()
