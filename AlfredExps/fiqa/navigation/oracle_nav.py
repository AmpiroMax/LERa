import argparse
import os
import json

import numpy as np
from collections import Counter, OrderedDict
from typing import Generator

from fiqa.navigation.basics_and_dummies import NavigatorBase
from fiqa.language_processing.subtask import Subtask
from fiqa.alfred_thor_env import AlfredThorEnv
from alfred_utils.gen.constants import (
    RECEPTACLES, MOVABLE_RECEPTACLES, OPENABLE_CLASS_SET, VISIBILITY_DISTANCE, 
    AGENT_ROTATE_ADJ, AGENT_HORIZON_ADJ
)

from utils.logger import logger


class OracleNavigator(NavigatorBase):
    """A navigator that teleports to receptacles and not hidden objects 
    in the current scene. The code is based on 
    https://github.com/alfworld/alfworld/blob/master/alfworld/agents/controller/oracle.py
    """

    def __init__(self, args: argparse.Namespace, env: AlfredThorEnv) -> None:
        super().__init__(args)
        self.env = env
        self.openable_points = None
        self.static_receps = OrderedDict()  # to ensure the same iteration order in 3.6

        self.target_objs_init_pos = dict()  # is useful for 'pick_two_...' tasks
        self.last_teleport_to_recep = None  # is useful for 'pick_two_...' tasks
        self.target_parent = None  # is useful for 'pick_two_...' tasks
        self.is_pick_two_task = False  # is useful for 'pick_two_...' tasks

        self.not_hidden_goal_objs = dict()  # is iterated at most once
        self.positions_source = None
        self.taken_from_alfred = False  # is useful when retry_nav == True

        self.teleport_action = None
        self.was_teleported = False

        RECEPTACLES_SB = set(RECEPTACLES) | {'Sink', 'Bathtub'}
        self.STATIC_RECEPTACLES = RECEPTACLES_SB - set(MOVABLE_RECEPTACLES)

        # TODO: add for valid_unseen...
        self.distance_fix = {
            'valid_seen': {
                'SideTable': {329,},
                'Drawer': {224,},
                'Fridge': {7,},
                'Microwave': {24,},
                'GarbageCan': {2, 20, 25, 418, 419},
                'ToiletPaperHanger': {406, 415},  # this is even an 'existence' fix
                'Toilet': {410, 417},  # this is even an 'existence' fix
            },
            'valid_unseen': dict()
        }
        self.horizon_fix = {
            'valid_seen': {
                'SideTable': {326,},
                'CounterTop': {14,},
                'DiningTable': {21, 28},
                'Drawer': {304,}
            },
            'valid_unseen': dict()
        }
        self.problem_receps = (
            set(self.distance_fix[args.split].keys())
            | set(self.horizon_fix[args.split].keys())
        )
        self.problem_scenes = set()
        for scene_source in [self.distance_fix[args.split], self.horizon_fix[args.split]]:
            for scenes in scene_source.values():
                self.problem_scenes.update(scenes)
        self.use_fix = False
        # There are also "bad objects" that our implementation cannot iteract
        # with because of AI2THOR quirks, so we just disregard them
        self.bad_objs = {
            'valid_seen': {
                'Drawer_a96cbd9d', 'ButterKnife_a7531534(Clone)_copy_35',
                'ButterKnife_628dcab4(Clone)_copy_42',
                'CD_b96b25c3(Clone)_copy_19', 'CreditCard_8cb9b9d4(Clone)_copy_10'
                # 'ButterKnife_8393cb4d' <-- we cannot disregard it, 
                # since this is the only unhidden knife (792-797 episodes fail)
            },
            'valid_unseen': set()
        }

        print(
            'WARNING! You are using OracleNavigator. Currently, '
            + 'it cannot navigate to objects hidden in a closed receptacle.'
        )

    def reset(self, traj_data: dict) -> None:
        self.openable_points = None
        self.static_receps = OrderedDict()

        self.is_pick_two_task = traj_data['task_type'] == 'pick_two_obj_and_place'
        if self.is_pick_two_task:
            self.last_teleport_to_recep = None
            self.target_parent = traj_data['pddl_params']['parent_target']
            target_obj = traj_data['pddl_params']['object_target']
            self.target_objs_init_pos = dict()
            for obj in self.env.last_event.metadata['objects']:
                if obj['objectType'] == target_obj:
                    self.target_objs_init_pos[obj['name']] = obj['position']

        # If the points are not suitable for the episode execution, signal it
        cur_scene_id = int(traj_data['root'].split('/')[3].split('-')[-1])
        self.use_fix = cur_scene_id in self.problem_scenes

        # Use pre-computed openable points from ALFRED to store receptacle locations
        scene_num = traj_data['scene']['scene_num']
        openable_json_file = os.path.join(
            'alfred_utils/gen/', f'layouts/FloorPlan{scene_num}-openable.json'
        )
        with open(openable_json_file, 'r') as f:
            self.openable_points = json.load(f)

        # Find all the static receptacles
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        for object_id, point in self.openable_points.items():
            action = {
                'action': 'TeleportFull',
                'x': point[0],
                'y': agent_height,
                'z': point[1],
                'rotateOnTeleport': False,
                'rotation': point[2],
                'horizon': point[3]
            }
            event = self.env.step(action)

            if event.metadata['lastActionSuccess']:
                instance_segs = self.env.last_event.instance_segmentation_frame
                color_to_object_id = self.env.last_event.color_to_object_id

                # Find unique instance segs
                color_count = Counter()
                for x in range(instance_segs.shape[0]):
                    for y in range(instance_segs.shape[1]):
                        # color = instance_segs[y, x]  <-- Bug in ALFWorld?
                        color = instance_segs[x, y]
                        color_count[tuple(color)] += 1

                for color, num_pixels in color_count.most_common():
                    if color in color_to_object_id:
                        object_id = color_to_object_id[color]
                        object_type = object_id.split('|')[0]
                        if 'Basin' in object_id:
                            object_type += 'Basin'

                        if object_type in self.STATIC_RECEPTACLES:
                            if object_id not in self.static_receps:
                                self.static_receps[object_id] = {
                                    'object_id': object_id,
                                    'object_type': object_type,
                                    'locs': action,
                                    'num_pixels': num_pixels,
                                    'num_id': f'{object_type.lower()} '
                                    + f'{str(self._next_num_id(object_type))}',
                                    'closed': True 
                                    if object_type in OPENABLE_CLASS_SET
                                    else None
                                }
                            elif object_id in self.static_receps and (
                                num_pixels > self.static_receps[object_id]['num_pixels']
                            ):
                                self.static_receps[object_id]['locs'] = action
                                self.static_receps[object_id]['num_pixels'] = num_pixels

    def _next_num_id(self, object_type: str) -> int:
        return len([
            obj for _, obj in self.static_receps.items() 
            if obj['object_type'] == object_type
        ]) + 1

    def reset_before_new_objective(self, subtask: Subtask, retry_nav: bool) -> None:
        self.teleport_action = None
        self.was_teleported = False

        if not retry_nav:
            self.taken_from_alfred = False
            if subtask.obj not in self.STATIC_RECEPTACLES:
                # Since movable objects can change their positions, 
                # it's necessary to obtain not hidden instances here, 
                # not in self.reset()
                self.not_hidden_goal_objs = dict()
                for obj in self.env.last_event.metadata['objects']:
                    if obj['objectType'] != subtask.obj:
                        continue
                    if (
                        self.is_pick_two_task 
                        and obj['name'] in self.target_objs_init_pos
                    ):
                        cur_pos = np.array(list(obj['position'].values()))
                        initial_pos = np.array(list(
                            self.target_objs_init_pos[obj['name']].values()
                        ))
                        pos_dif = np.linalg.norm(cur_pos - initial_pos)
                        if pos_dif > 1e-3:
                            # We've chosen 1e-3 since some objects can 
                            # change coordinates by 1e-4 --- 1e-5 even if they
                            # weren't touched (we observed this with KeyChain)
                            continue
                    hidden = False
                    if obj['parentReceptacles'] is not None:
                        for recep in obj['parentReceptacles']:
                            # If the subtasks are correct and the execution 
                            # is also correct, any previously opened receptacle 
                            # must be closed before a nav subtask
                            if recep.split('|')[0] in OPENABLE_CLASS_SET:
                                hidden = True
                                break
                    if not hidden:
                        self.not_hidden_goal_objs[obj['name']] = obj

                if len(self.not_hidden_goal_objs) == 0:
                    msg = 'Oracle nav has not found an instance of ' \
                        + f'{subtask.obj} that is not hidden'
                    logger.log_error(msg)
                    assert False, msg
                self.positions_source = self._get_interactable_positions()
            elif (
                self.is_pick_two_task
                and self.last_teleport_to_recep is not None
                and self.target_parent == subtask.obj
            ):
                self.positions_source = iter((self.last_teleport_to_recep,))
                # Although self.last_teleport_to_recep can be obtained from
                # self._get_position_source_for_static_receps(), we say that it was taken
                # from ALFRED because later the position can be fixed
                # (when retry_nav == True)
                self.taken_from_alfred = True
            elif self.use_fix and subtask.obj in self.problem_receps:
                self.positions_source = (
                    self._get_position_source_for_static_receps(subtask.obj)
                )
            else:
                self.positions_source = (
                    obj['locs'] for _, obj in self.static_receps.items()
                    if subtask.obj == obj['object_type']
                )
                self.taken_from_alfred = True
        else:  # retry_nav == True
            if self.args.allow_retry_nav_for_oracle_nav:
                if self.taken_from_alfred:
                    self.positions_source = (
                        self._get_position_source_for_static_receps(subtask.obj)
                    )
                    self.taken_from_alfred = False
            else:
                self.positions_source = iter([])
            # if self.args.planner != 'with_replan' and self.taken_from_alfred:
            #     self.positions_source = (
            #         self._get_position_source_for_static_receps(subtask.obj)
            #     )
            #     self.taken_from_alfred = False
            # elif (
            #     self.args.planner == 'with_replan'
            #     and subtask.obj in self.STATIC_RECEPTACLES
            # ):
            #     self.positions_source = iter([])

        try:
            self.teleport_action = next(self.positions_source)
            if (
                self.is_pick_two_task 
                and subtask.obj in self.STATIC_RECEPTACLES
                and subtask.obj == self.target_parent
            ):
                self.last_teleport_to_recep = self.teleport_action
        except StopIteration:
            self.teleport_action = None
        
        if self.teleport_action is None:
            if self.args.planner == 'with_replan':
                self.teleport_action = {'action': 'StopNav'}  # Refuse to navigate
            else:
                msg = 'Oracle nav has not found a valid position for ' \
                    + f'{subtask.obj}!'
                logger.log_error(msg)
                assert False, msg

    def _get_position_source_for_static_receps(
        self, recep_class: str
    ) -> Generator[dict, None, None]:
        self.not_hidden_goal_objs = dict()  # here "objs" <=> "receps"
        for obj in self.env.last_event.metadata['objects']:
            if obj['objectType'] != recep_class:
                continue
            self.not_hidden_goal_objs[obj['name']] = obj
        return self._get_interactable_positions()

    def _get_interactable_positions(self) -> Generator[dict, None, None]:
        event = self.env.step(action={'action': 'GetReachablePositions'})
        reachable_positions = event.metadata['actionReturn']

        for _, obj in self.not_hidden_goal_objs.items():
            if obj['name'] in self.bad_objs[self.args.split]:
                continue
            # Firstly, get all positions within VISIBILITY_DISTANCE 
            # and sort them by distance to the goal object
            le_vis_distance_positions = []
            obj_pos = obj['position']
            for pos in reachable_positions:
                distance = (
                    (pos['x'] - obj_pos['x']) ** 2 
                    + (pos['y'] - obj_pos['y']) ** 2
                    + (pos['z'] - obj_pos['z']) ** 2
                ) ** 0.5
                # We noticed there are some cases when the instance 
                # is considered by AI2THOR as not visible, although
                # the distance is <= 1.5m. Therefore, we decided to make 
                # the condition sticter: < 1.45m
                if distance < VISIBILITY_DISTANCE - 0.05:
                    le_vis_distance_positions.append({'pos': pos, 'distance': distance})
            le_vis_distance_positions = sorted(
                le_vis_distance_positions, key=lambda x: x['distance']
            )
            # Secondly, shrink the number of positions, since going through all
            # positions can take too long (2.5-3.5 minutes for an instance)
            le_vis_distance_positions = le_vis_distance_positions[:10]
            # Finally, go through all reasonable angles of view and select 
            # the view with the max binary mask of the goal object
            interactable_positions = dict()
            for pos_and_dist in le_vis_distance_positions:
                pos = pos_and_dist['pos']
                max_mask_size = 0
                best_yaw = 0
                best_horizon = 0
                for yaw_iter in range(4):
                    yaw = yaw_iter * AGENT_ROTATE_ADJ
                    # We decided that the range [-30, 60] is more than enough
                    for horizon_iter in range(7):
                        horizon = 60 - horizon_iter * AGENT_HORIZON_ADJ
                        self.env.step(action={
                            'action': 'TeleportFull',
                            'x': pos['x'],
                            'y': pos['y'],  # equals agent's height
                            'z': pos['z'],
                            'rotation': yaw,
                            'horizon': horizon
                        })
                        mask_max_size = self._find_mask_max_size(obj['objectType'])
                        if mask_max_size > max_mask_size:
                            max_mask_size = mask_max_size
                            best_yaw = yaw
                            best_horizon = horizon
                if max_mask_size == 0:
                    continue
                pos_with_angles = {
                    'object_type': obj['objectType'],
                    'max_mask_size': max_mask_size,
                    'locs': {
                        'action': 'TeleportFull',
                        'x': pos['x'],
                        'y': pos['y'],
                        'z': pos['z'],
                        'rotation': best_yaw,
                        'horizon': best_horizon
                    }
                }
                dist = pos_and_dist['distance']
                interactable_positions[obj['name'] + f'{dist: .6f}'] = pos_with_angles

            if len(interactable_positions) == 0:
                msg = 'Oracle nav has not found an interactable position for' \
                    + f' an instance of {obj["objectType"]} that is not hidden'
                logger.log_warning(msg)
                continue
            # Since ALFRED only allows 9 interact API fails, 
            # shrink the number of positions in case of multiple instances
            new_len = 10 // len(self.not_hidden_goal_objs)
            if new_len < 3:
                new_len = 3
            if self.args.planner == 'with_replan' and new_len > 3:
                # Since objs like Fridge are often in the number of one instance and
                # since replanning is enabled, new_len == 10 is not acceptable
                new_len = 3
            interactable_positions = list(sorted(
                interactable_positions.values(), 
                key=lambda x: x['max_mask_size'],
                reverse=True
            ))[:new_len]
            for pos in interactable_positions:
                yield pos['locs']

    def _find_mask_max_size(self, goal_obj_type: str) -> int:
        instance_masks = self.env.last_event.instance_masks
        mask_max_size = 0
        for obj_id, obj_mask in instance_masks.items():
            condition = False
            if (
                'Sliced' in goal_obj_type or goal_obj_type == 'SinkBasin' 
                or goal_obj_type == 'BathtubBasin'
            ):
                condition = goal_obj_type in obj_id
            else:
                obj_type = obj_id.split('|')[0]
                condition = goal_obj_type == obj_type

            if condition:
                mask_cur_size = np.sum(obj_mask)
                min_pixel_depth = np.min(
                    self.env.last_event.depth_frame[obj_mask]
                ) / 1000.0  # mm -> m
                if (
                    mask_cur_size > mask_max_size
                    and min_pixel_depth < VISIBILITY_DISTANCE - 0.05
                ):
                    mask_max_size = mask_cur_size
        return mask_max_size

    def __call__(self, rgb: np.ndarray) -> dict:
        if self.was_teleported:
            return {'action': 'StopNav'}
        self.was_teleported = True
        return self.teleport_action
