"""
The script is heavily based on: https://github.com/soyeonm/FILM 
"""
import argparse
import cv2
import copy
import math
import os
import glob
# import random
# import gen.constants as constants
import numpy as np
from collections import OrderedDict
# from env.tasks import get_task
# from ai2thor.controller import Controller
# import gen.utils.image_util as image_util
# from gen.utils import game_util
# from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj
# from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
import skimage.morphology
# import yaml
# import sys

# from alfred_utils.env.thor_env import ThorEnv
# from alfred_utils.env.thor_env_code import ThorEnvCode

# import pandas as pd
# import json
from typing import List, Optional, Tuple
# from types import SimpleNamespace
# from allenact.utils.cache_utils import DynamicDistanceCache
import torch
import torch.nn as nn
# from torchvision import transforms
import torchvision.transforms as T

from fiqa.navigation.film.alfred_perception_models \
    import AlfredSegmentationAndDepthModel
# from fiqa.perceivers.maskrcnn.alfworld_mrcnn import load_pretrained_model 
# import fiqa.perceivers.maskrcnn.alfworld_constants as alfworld_constants
import fiqa.navigation.film.envs.utils.pose as pu
# from fiqa.navigation.film.ALFRED_task_helper \
#     import get_list_of_highlevel_actions

# import envs.utils.pose as pu
import fiqa.navigation.film.utils_f.control_helper as CH
from fiqa.navigation.film.envs.utils.fmm_planner import FMMPlanner
from fiqa.navigation.film.sem_mapping import SemanticMapping
from fiqa.navigation.film.sem_map_model import UNetMulti

from fiqa.navigation.basics_and_dummies import NavigatorBase
from fiqa.task_handlers.info_retriever import InfoRetrieverBase
from fiqa.perceivers.basics_and_dummies import SegModelBase
from fiqa.language_processing.subtask import Subtask


class FILMNavigator(NavigatorBase):
    """The navigator from the FILM article: https://arxiv.org/abs/2110.07342
    Uses the semantic mapping module, the semantic search module and
    the deterministic policy (based on FMM method) to explore a scene and
    then to reach objects.

    N.B. It would be too laborious to fully rewrite the code that's why 
    the implementation lacks clarity, documentation, and good structure...
    """

    def __init__(
        self, args: argparse.Namespace, seg_tuple: Tuple[SegModelBase, T.Compose]
    ) -> None:
        super().__init__(args)
        path_prefix = 'fiqa/checkpoints/Pretrained_Models_FILM/'

        self.seg_model, self.img_transform_seg = seg_tuple

        # TODO: move it to the depth module
        # INIT DEPTH MODEL
        ##################################################################################
        self.depth_gpu = torch.device(
            f'cuda:{args.navigator_gpu}' if torch.cuda.is_available() else 'cpu'
        )
        model_path = (
            path_prefix + 'depth_models/valts/model-2000-best_silog_10.13741'
        )  # 45 degrees only model
        state_dict = torch.load(model_path, map_location=self.depth_gpu)['model']

        self.depth_pred_model = AlfredSegmentationAndDepthModel()
        new_checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]; new_checkpoint[name] = v
        state_dict = new_checkpoint; del new_checkpoint

        self.depth_pred_model.load_state_dict(state_dict)
        self.depth_pred_model.eval()
        self.depth_pred_model.to(device=self.depth_gpu)

        model_path = path_prefix + 'depth_models/valts0/model-102500-best_silog_17.00430'
        self.depth_pred_model_0 = AlfredSegmentationAndDepthModel()
        state_dict = torch.load(model_path, map_location=self.depth_gpu)['model']

        new_checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]; new_checkpoint[name] = v
        state_dict = new_checkpoint; del new_checkpoint

        self.depth_pred_model_0.load_state_dict(state_dict)
        self.depth_pred_model_0.eval()
        self.depth_pred_model_0.to(device=self.depth_gpu)
        ##################################################################################

        # if not args.film_use_oracle_seg:
            # SEGMENTATION MODEL 
            ######################################################
            # #LARGE
            # self.sem_seg_model_alfw_large = load_pretrained_model(
            #     path_prefix + 'maskrcnn_alfworld/receps_lr5e-3_003.pth', 
            #     self.depth_gpu, 'recep'
            # )
            # self.sem_seg_model_alfw_large.eval()
            # self.sem_seg_model_alfw_large.to(self.depth_gpu)

            # self.large = alfworld_constants.STATIC_RECEPTACLES
            # self.large_objects2idx = {k:i for i, k in enumerate(self.large)}
            # self.large_idx2large_object = {v:k for k,v in self.large_objects2idx.items()}

            # #SMALL    
            # self.sem_seg_model_alfw_small = load_pretrained_model(
            #     path_prefix + 'maskrcnn_alfworld/objects_lr5e-3_005.pth', 
            #     self.depth_gpu, 'obj'
            # )
            # self.sem_seg_model_alfw_small.eval()
            # self.sem_seg_model_alfw_small.to(self.depth_gpu)

            # self.small = alfworld_constants.OBJECTS_DETECTOR
            # self.small_objects2idx = {k:i for i, k in enumerate(self.small)}
            # self.small_idx2small_object = {v:k for k,v in self.small_objects2idx.items()}

        self.map_save_large_objects = [
            'ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 
            'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 
            'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 
            'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 
            'Sofa', 'StoveBurner', 'TVStand', 'Toilet'
        ]  # is taken from FILM's constants, len() == 24
        self.map_all_objects = [
            'AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 
            'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 
            'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 
            'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 
            'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 
            'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 
            'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 
            'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 
            'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 
            'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 
            'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 
            'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 
            'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 
            'WineBottle'
        ]  # is taken from FILM's constants, len() == 73

        # self.res = transforms.Compose([
        #     transforms.ToPILImage(), 
        #     transforms.Resize((150, 150), interpolation=Image.NEAREST)
        # ])

        self.info = {}
        # self.info['distance_to_goal'] = None
        # self.info['spl'] = None
        # self.info['success'] = None
        self.total_cat2idx = {}

        self.transform = T.Compose([T.ToTensor()])

        # self.event = self.reset_nav()
        self.map_gpu = torch.device(
            f'cuda:{args.navigator_gpu}' if torch.cuda.is_available() else 'cpu'
        )
        self.Unet_model = UNetMulti(
            (240, 240), num_sem_categories=24
        ).to(device=self.map_gpu)
        sd = torch.load(path_prefix + 'new_best_model.pt', map_location=self.map_gpu)
        self.Unet_model.load_state_dict(sd)
        self.softmax = nn.Softmax(dim=1)

        self.prev_rgb = None
        self.goal_name = None
        self.goal_idx = None
        self.cat_equate_dict = {}
        self.newly_goal_set = False
        self.global_goals = []
        self.goal_spotted_s = [False,]
        self.found_goal = [0,]  # it seems to be equal to goal_spotted_s
        self.prev_wall_goal = None
        self.dilation_deg = 0
        self.camera_horizon = 30  # In all ALFRED train-split scenes, the initial angle is set to 30
        self.o = 0.0
        self.all_objects2idx = {o: i for i, o in enumerate(self.map_all_objects)}
        self.mask_new = None
        self.map_save_large_objects2idx = {
            obj: i for i, obj in enumerate(self.map_save_large_objects)
        }
        self.picked_up = False
        self.picked_up_cat = None  # Is not used
        self.picked_up_mask = None
        self.prev_number_action = None
        self.consecutive_steps = False
        self.final_sidestep = False
        self.pose_correction_type = None
        self.is_moving_behind = False
        self.in_move_behind_correction = False
        self.move_behind_progress = 0
        self.last_three_sidesteps = [None,] * 3
        self.side_step_order = 0
        self.rotate_aftersidestep = None
        self.prev_sidestep_success = None
        self.prev_sidestep_success_needs_update = False
        self.doing_side_step_order_1_2 = False
        self.side_step_helper_save = dict()
        self.opp_side_step = False
        self.in_left_side_step_correction = False
        self.in_right_side_step_correction = False
        self.move_until_visible_order = 0
        self.mvb_num_in_cycle = 0
        self.mvb_which_cycle = 0
        self.move_until_visible_cycled = False
        self.move_until_visible_helper_progress = 0
        self.in_move_until_visible_helper = False
        self.in_move_until_visible_correction = False
        self.delete_lamp = False
        self.in_plan_act_else = False
        self.plan_act_else_save = dict()
        self.learned_depth_frame = None
        self.interaction_mask = None
        self.execute_interaction = False
        self.caution_pointers = set()
        self.last_not_nav_subtask = None
        self.last_not_nav_subtask_success = None
        self.cur_nav_subtask = None
        self.is_pick_two_objs_and_place_task = False
        self.obj_class_in_pick_two_task = None
        self.recep_class_in_pick_two_task = None

        # Restore FILM's args (they are taken from the original FILM's agent):
        self.args.num_sem_categories = 1
        self.args.obstacle_selem = 4
        self.args.num_processes = 1
        self.args.device = self.map_gpu
        self.args.frame_height = 150
        self.args.frame_width = 150
        self.args.map_resolution = 5
        self.args.map_size_cm = 1200
        self.args.global_downscaling = 1
        self.args.vision_range = 100
        self.args.hfov = 60.0
        self.args.du_scale = 1
        self.args.print_time = 0
        self.args.cat_pred_threshold = 10
        self.args.exp_pred_threshold = 1.0
        self.args.map_pred_threshold = 65
        self.args.no_straight_obs = True
        self.args.camera_height = 1.55
        self.args.env_frame_width = 300
        self.args.delete_from_map_after_move_until_visible = True
        self.args.num_local_steps = 25
        self.args.delete_pick2 = False  # It is not clear what this is. Not actually used.
        self.args.approx_error_message = True
        self.args.no_rotate_sidestep = False
        self.args.no_opp_sidestep = False
        self.args.check_before_sidestep = True
        self.args.collision_threshold = 0.2
        self.args.step_size = 5
        self.args.use_sem_policy = True
        self.args.stop_cond = 0.55
        self.args.no_pickup = False
        self.args.no_pickup_update = True
        self.args.side_step_step_size = 3
        self.args.sidestep_width = 0
        # Although `args.no_delete_lamp` is set to False in FILM, we see it as the reason
        # for the infinite loop with 'LookUp_0' here, so we disable the deletion
        self.args.no_delete_lamp = True
        self.use_stop_analysis = args.film_use_stop_analysis

        self.selem = skimage.morphology.square(self.args.obstacle_selem)
        self.steps_taken = 0
        self.steps_taken_before_new_obj = 0
        self.number_of_plan_act_calls = 0  # <==> FILM's self.steps from sem_exp_thor.py

        self.wheres_delete_s = [np.zeros((240, 240))] * self.args.num_processes
        self.do_not_update_cat_s = [None] * self.args.num_processes
        self.num_scenes = self.args.num_processes  # == 1
        self.last_action_ogn = "<<stop>>"
        self.last_action = 'StopNav'
        self.action_5_count = 0
        self.lookDownUpLeft_count = 0
        self.cam_target_angle = None  # is needed for set_back_to_angle()
        self.is_changing_cam_angle = False  # is needed for set_back_to_angle()

    def reset(self, subtask_queue: List[Subtask]) -> None:
        self.frames = []

        if self.args.draw_debug_imgs:
            self.dir_idx = len(glob.glob('debug_imgs/*'))
            os.makedirs(f'debug_imgs/{self.dir_idx}', exist_ok=True)

        # task_id, r_idx = traj_data['task_id'], traj_data['repeat_idx']
        # task_key = (task_id, r_idx)
        # subtask_queue = self.instructions_processed[task_key]
        self.cats_in_subtask_queue = self._get_cats_from_subtasks(subtask_queue)
        self.reset_total_cat_new(self.cats_in_subtask_queue)
        self.nc = self.args.num_sem_categories + 4  # num channels
        # Strictly speaking, this implementation of self.caution_pointers is
        # used in a wider range of cases than FILM's, but "extra"
        # cases should be rare.
        self.caution_pointers = set()
        for i, subtask in enumerate(subtask_queue):
            if subtask.action == 'PutObject':
                j = i - 1
                while j >= 0:
                    if subtask_queue[j].action == 'GotoLocation':
                        self.caution_pointers.add((
                            subtask_queue[j].action, subtask_queue[j].obj,
                            subtask_queue[j].recept
                        ))
                        break
                    j -= 1
        # Determine whether the task is pick_two_objs_and_place_task
        # Warning: if the language module makes an error, the check is false
        self.is_pick_two_objs_and_place_task = False
        self.obj_class_in_pick_two_task = None
        self.recep_class_in_pick_two_task = None
        all_pickups_and_puts_with_goto = [
            subtask for subtask in subtask_queue
            if subtask.action in ['GotoLocation', 'PickupObject', 'PutObject']
        ]  # Delete potential open/close subtasks
        if len(all_pickups_and_puts_with_goto) > 2:
            obj_class = all_pickups_and_puts_with_goto[0].obj
            recep_class = all_pickups_and_puts_with_goto[2].obj
        else:
            obj_class = None
            recep_class = None
        if (
            len(all_pickups_and_puts_with_goto) > 7
            and all_pickups_and_puts_with_goto[0].action == 'GotoLocation'
            and all_pickups_and_puts_with_goto[1].action == 'PickupObject'
            and all_pickups_and_puts_with_goto[2].action == 'GotoLocation'
            and all_pickups_and_puts_with_goto[3].action == 'PutObject'
            and all_pickups_and_puts_with_goto[4].action == 'GotoLocation'
            and all_pickups_and_puts_with_goto[4].obj == obj_class
            and all_pickups_and_puts_with_goto[5].action == 'PickupObject'
            and all_pickups_and_puts_with_goto[5].obj == obj_class
            and all_pickups_and_puts_with_goto[6].action == 'GotoLocation'
            and all_pickups_and_puts_with_goto[6].obj == recep_class
            and all_pickups_and_puts_with_goto[7].action == 'PutObject'
            and all_pickups_and_puts_with_goto[7].obj == recep_class
        ):
            self.is_pick_two_objs_and_place_task = True
            self.obj_class_in_pick_two_task = obj_class
            self.recep_class_in_pick_two_task = recep_class

        # Strictly speaking, our self.steps_taken differs from FILM's
        # because it doesn't account for interactions, but interaction actions
        # are rare compared to navigation actions, so our self.steps_taken
        # affects navigator's behaviour approximately equivalently
        self.steps_taken = 0
        self.number_of_plan_act_calls = 0
        self.action_5_count = 0
        self.camera_horizon = 30
        self.lookDownUpLeft_count = 0
        self.cam_target_angle = None
        self.is_changing_cam_angle = False
        self.picked_up = False
        self.picked_up_mask = None
        self.consecutive_steps = False
        self.final_sidestep = False
        self.is_moving_behind = False
        self.move_behind_progress = 0
        self.last_three_sidesteps = [None,] * 3
        self.side_step_order = 0
        self.rotate_aftersidestep = None
        self.prev_sidestep_success = None
        self.prev_sidestep_success_needs_update = False
        self.doing_side_step_order_1_2 = False
        self.move_until_visible_order = 0
        self.move_until_visible_helper_progress = 0
        self.in_move_until_visible_helper = False
        self.wheres_delete_s[0] = np.zeros((240, 240))
        self.do_not_update_cat_s = [None]
        self.in_plan_act_else = False
        self.interaction_mask = None
        self.execute_interaction = False
        self.last_not_nav_subtask = None
        self.last_not_nav_subtask_success = None
        self.pose_correction_type = None

        self.cat_equate_dict = {}
        action_and_obj = set(
            (subtask.action, subtask.obj) for subtask in subtask_queue
        )
        # Since our language model doesn't always predict 'Knife' for slicing
        # and 'FloorLamp' for examining in light,
        # the class equating conditions were changed
        if ('ToggleObjectOn', 'DeskLamp') in action_and_obj:
            self.total_cat2idx['FloorLamp'] = self.total_cat2idx['DeskLamp']
            self.cat_equate_dict['FloorLamp'] = 'DeskLamp'
        elif ('ToggleObjectOn', 'FloorLamp') in action_and_obj:
            self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp']
            self.cat_equate_dict['DeskLamp'] = 'FloorLamp'
        if 'SliceObject' in (act for act, _ in action_and_obj):
            if ('PickupObject', 'Knife') in action_and_obj:
                self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife']
                self.cat_equate_dict['ButterKnife'] = 'Knife'
            elif ('PickupObject', 'ButterKnife') in action_and_obj:
                self.total_cat2idx['Knife'] = self.total_cat2idx['ButterKnife']
                self.cat_equate_dict['Knife'] = 'ButterKnife'

        self.seg_model.cat_equate_dict = self.cat_equate_dict
        self.seg_model.total_cat2idx = self.total_cat2idx
        self.args.num_sem_categories = len(self.total_cat2idx)
        self.sem_map_module = SemanticMapping(self.args).to(self.map_gpu)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles([45] * 1)
        # self.info['sensor_pose'] = [0., 0., 0.]  # is done in reset_nav()
        self.reset_nav()
        self.reset_map()
        # self.init_map_and_pose()  # <-- we don't need this, since we have only one "environment"
        self.init_map_and_pose_for_env(0)
        self.last_action_ogn = "<<stop>>"
        self.last_action = 'StopNav'
        self.goal_spotted_s = [False,]
        self.found_goal = [0,]

    def update_state(self, update_info: dict) -> None:
        cats_in_subtask_queue = set(self.cats_in_subtask_queue)
        cats_in_new_subtask_queue = set(
            self._get_cats_from_subtasks(update_info['subtask_queue'])
        )
        cats_diff = cats_in_new_subtask_queue - cats_in_subtask_queue
        assert len(cats_diff) == 0, \
            'Error! Updating objects is not currently supported! New objects: ' + f"{cats_diff}"

        self.caution_pointers = set()
        new_subtask_queue = update_info['subtask_queue']
        for i, subtask in enumerate(new_subtask_queue):
            if subtask.action == 'PutObject':
                j = i - 1
                while j >= 0:
                    if new_subtask_queue[j].action == 'GotoLocation':
                        self.caution_pointers.add((
                            new_subtask_queue[j].action, new_subtask_queue[j].obj,
                            new_subtask_queue[j].recept
                        ))
                        break
                    j -= 1
        
        if update_info['task_type'] == 'pick_two_obj_and_place':
            self.is_pick_two_objs_and_place_task = True
            self.obj_class_in_pick_two_task = None
            self.recep_class_in_pick_two_task = None
            for subtask in new_subtask_queue:
                if self.obj_class_in_pick_two_task is None and subtask.action == 'PickupObject':
                    self.obj_class_in_pick_two_task = subtask.obj
                if self.recep_class_in_pick_two_task is None and subtask.action == 'PutObject':
                    self.recep_class_in_pick_two_task = subtask.obj

        self.cat_equate_dict = {}
        action_and_obj = set(
            (subtask.action, subtask.obj) for subtask in new_subtask_queue
        )
        if ('ToggleObjectOn', 'DeskLamp') in action_and_obj:
            self.total_cat2idx['FloorLamp'] = self.total_cat2idx['DeskLamp']
            self.cat_equate_dict['FloorLamp'] = 'DeskLamp'
        elif ('ToggleObjectOn', 'FloorLamp') in action_and_obj:
            self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp']
            self.cat_equate_dict['DeskLamp'] = 'FloorLamp'
        elif 'SliceObject' in (act for act, _ in action_and_obj):
            if ('PickupObject', 'Knife') in action_and_obj:
                self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife']
                self.cat_equate_dict['ButterKnife'] = 'Knife'
            elif ('PickupObject', 'ButterKnife') in action_and_obj:
                self.total_cat2idx['Knife'] = self.total_cat2idx['ButterKnife']
                self.cat_equate_dict['Knife'] = 'ButterKnife'
        else:
            self.cat_equate_dict = dict()
            for i, key in enumerate(self.total_cat2idx.keys()):
                self.total_cat2idx[key] = i  # Relies on the constant order of the keys
            self.goal_idx2cat = {v:k for k, v in self.total_cat2idx.items()}
        self.seg_model.cat_equate_dict = self.cat_equate_dict
        self.seg_model.total_cat2idx = self.total_cat2idx
        self.args.num_sem_categories = len(self.total_cat2idx)

    def reset_before_new_objective(
        self, subtask: Subtask, retry_nav: bool, pose_correction_type: Optional[str] = None
    ) -> None:
        self.steps_taken_before_new_obj = self.steps_taken
        self.prev_number_action = None  # In order to enable calling _plan()
        self.info['sensor_pose'] = [0., 0., 0.]  # Interaction doesn't change the pose
        if not retry_nav:
            self.goal_name = subtask.obj
            self.goal_idx = self.total_cat2idx[self.goal_name]
            self.cur_nav_subtask = (subtask.action, subtask.obj, subtask.recept)
            self.prev_wall_goal = None
            self.dilation_deg = 0
            self.goal_spotted_s = [False,]
            self.found_goal = [0,]
            self.mvb_num_in_cycle = 0
            self.mvb_which_cycle = 0
            self.pose_correction_method = None
        else:
            if pose_correction_type is not None:
                assert pose_correction_type in [
                    'move_behind', 'right_side_step', 'left_side_step',
                    'move_until_visible'
                ], f'Not implemented correction method {pose_correction_type}!'
                self.pose_correction_type = pose_correction_type
        if abs(self.camera_horizon - 45) > 5:
            self.is_changing_cam_angle = True
            self.cam_target_angle = 45

    def __call__(self, rgb: np.ndarray) -> str:
        if self.steps_taken > self.steps_taken_before_new_obj:
            whether_success = self.get_approximate_success(rgb)
            if whether_success:
                if self.last_action == 'LookUp':
                    self.camera_horizon -= 15
                elif self.last_action == 'LookDown':
                    self.camera_horizon += 15
            else:
                if (
                    self.last_action == 'LookUp'
                    or self.last_action == 'LookDown'
                ):  # In order to escape the loop of failed LookUp/LookDown
                    self.is_changing_cam_angle = False
                    self.cam_target_angle = None
            if self.prev_sidestep_success_needs_update:
                self._side_step_order_1_helper(whether_success)
                self.prev_sidestep_success_needs_update = False
            
            dx, dy, do = self.get_pose_change_approx_relative(
                self.last_action, whether_success
            )
            if self.consecutive_steps:
                self.accumulated_pose += np.array([dx, dy, do])
                if self.accumulated_pose[2] >= np.pi - 1e-1:
                    self.accumulated_pose[2] -= 2 * np.pi
                # Added by ourselves to match self.o
                elif self.accumulated_pose[2] <= -(2 * np.pi - 1e-1):
                    self.accumulated_pose[2] += 2 * np.pi
                if (
                    self.final_sidestep
                    and (
                        self.cam_target_angle is None  # LookUp/LookDown failed or was not used
                        or abs(self.camera_horizon - self.cam_target_angle) < 1  # LookUp/LookDown succeeded
                    )
                ):
                    # self.info['sensor_pose'] = copy.deepcopy(self.accumulated_pose).tolist()
                    dx, dy, do = copy.deepcopy(self.accumulated_pose).tolist()
                    self.info['sensor_pose'] = [abs(dx + dy), 0., do]  # <-- added by ourselves
                    if (
                        self.cam_target_angle is not None
                        or self.last_action not in ['LookUp', 'LookDown']
                    ):  # LookUp/LookDown succeeded or was not used
                        self.accumulated_pose = np.array([0.0, 0.0, 0.0])
                    # self.o = 0.0  # <-- is a strange moment in FILM...
                    self.final_sidestep = False
                    self.consecutive_steps = False
            else:
                self.info['sensor_pose'] = [abs(dx + dy), 0., do]  # <=> get_pose_change_approx()
            # self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        if (
            abs(self.camera_horizon) < 5 or abs(self.camera_horizon - 45) < 5
            and not self.consecutive_steps
        ):
            self.sem_seg_pred = self.get_sem_pred(rgb.copy())

            self.include_mask = np.sum(self.sem_seg_pred, axis=2).astype(bool).astype(float)
            self.include_mask = np.expand_dims(np.expand_dims(self.include_mask, 0), 0)
            self.include_mask = torch.tensor(self.include_mask).to(self.depth_gpu)

            self.depth = self.depth_pred_later(self.include_mask, rgb)
            self.depth = self._preprocess_depth(self.depth, 0., 50.)
            # TODO: do we need this?
            self.depth = np.nan_to_num(self.depth, nan=100, posinf=100, neginf=100)

            self.update_map(rgb)

        if self.pose_correction_type is not None:
            setattr(self, 'in_' + self.pose_correction_type + '_correction', True)
            if 'move_until_visible' == self.pose_correction_type:
                action = self.move_until_visible_correction(rgb=rgb.copy())
            else:
                action = getattr(self, self.pose_correction_type + '_correction')()
            self.pose_correction_type = None
        elif self.is_changing_cam_angle:
            action = self.set_back_to_angle(self.cam_target_angle)
            if action == 'LookUp_0':
                self.is_changing_cam_angle = False
                self.cam_target_angle = None
        elif self.is_moving_behind:
            action = self.move_behind()
        elif self.in_move_behind_correction:
            action = 'StopNav'
            self.in_move_behind_correction = False
        elif self.in_move_until_visible_helper:
            action = self.move_until_visible_helper()
        elif self.in_move_until_visible_correction:
            action = self.move_until_visible_correction()
        elif (
            (self.in_left_side_step_correction or self.in_right_side_step_correction)
            and self.side_step_order == 0
        ):  # Finished side_stepping
            action = 'StopNav'
            self.in_left_side_step_correction = self.in_right_side_step_correction = False
        else:
            if self.in_plan_act_else:
                self.plan_act_else_helper(rgb.copy())
                self.in_plan_act_else = False
            elif self.doing_side_step_order_1_2:
                self.side_step_helper(rgb.copy())
                self.doing_side_step_order_1_2 = False
            action = self.plan_act()

        while action == 'LookUp_0':  # Does only 1 iteration almost surely
            print('inside LookUp_0 loop')
            if self.in_plan_act_else:
                self.plan_act_else_helper(rgb.copy())
                self.in_plan_act_else = False
            elif self.doing_side_step_order_1_2:
                self.side_step_helper(rgb.copy())
                self.doing_side_step_order_1_2 = False
            action = self.plan_act()
        print('outside LookUp_0 loop')

        self.last_action = action = action.split('_')[0]
        self.steps_taken += 1
        self.prev_rgb = rgb

        if (
            self.args.draw_debug_imgs
            and (abs(self.camera_horizon) < 5 or abs(self.camera_horizon - 45) < 5)
        ):
            f = plt.figure(figsize=(16, 5))
            ax1 = f.add_subplot(1, 5, 1)
            ax2 = f.add_subplot(1, 5, 2)
            ax3 = f.add_subplot(1, 5, 3)
            ax4 = f.add_subplot(1, 5, 4)
            ax5 = f.add_subplot(1, 5, 5)
            ax1.imshow(rgb)
            ax1.set_title('RGB')
            ax2.imshow(self.sem_seg_pred[:, :, self.goal_idx])
            ax2.set_title(f'{self.goal_name} mask')
            # ax2.imshow(self.collision_map + self.local_map[0, 2].cpu().numpy())
            # ax2.set_title('Collision map')
            agent_loc = np.nonzero(self.local_map[0, 2].cpu().numpy()[:, ::-1])
            ax3.imshow(
                self.local_map[0, 0].cpu().numpy()[:, ::-1]
                + (self.mask_new[:, ::-1] if self.mask_new is not None else 0.)
            )
            xy = min(agent_loc[1]), min(agent_loc[0])  # transpose, because x axis has to be horizontal
            rect = patches.Rectangle((xy[0] - 1, xy[1] - 1), 3, 3, facecolor='r')
            ax3.add_patch(rect)
            goal_pos = (self.local_w - self.closest_goal[1], self.closest_goal[0])
            # goal_pos = self.closest_goal
            ax3.add_patch(patches.Circle(goal_pos, 3, color='w'))
            ax3.set_title('Obstacle map, agent, goal')
            ax3.set_xlabel('')
            ax3.set_ylabel('')
            ax4.imshow(
                self.local_map[0, 4 + self.goal_idx].cpu().numpy()[:, ::-1]
                + self.local_map[0, 2].cpu().numpy()[:, ::-1]
            )
            # ax4.add_patch(patches.Rectangle(xy, 5, -5, facecolor='r'))
            ax4.set_title(f'Sem. map for {self.goal_name}')
            ax5.imshow(self.fmm_dist[:, ::-1])
            rect = patches.Rectangle((xy[0] - 1, xy[1] - 1), 3, 3, facecolor='r')
            ax5.add_patch(rect)
            ax5.set_title('fmm_dist')
            img_idx = len(glob.glob(f'debug_imgs/{self.dir_idx}/*.png'))
            plt.savefig(f'debug_imgs/{self.dir_idx}/img_{img_idx}.png')
            plt.close(f)

        return action

    def _get_cats_from_subtasks(
        self, subtask_queue: List[Subtask]
    ) -> List[str]:
        return list(set(subtask.obj for subtask in subtask_queue))

    def reset_total_cat_new(self, categories_in_inst):
        self.total_cat2idx = {}
        # total_cat2idx["Knife"] = len(total_cat2idx)
        # total_cat2idx["SinkBasin"] = len(total_cat2idx)
        # if self.args.use_sem_policy:
        for obj in self.map_save_large_objects:
            # if not(obj == "SinkBasin"):
            self.total_cat2idx[obj] = len(self.total_cat2idx)

        # start_idx = len(total_cat2idx)  # 1 for "fake"
        # start_idx += 4 * 0  # self.rank == 0
        # cat_counter = 0
        # assert len(categories_in_inst) <= 6 + 1  # +1 since we add 'Knife'/'ButterKnife'
        # Keep total_cat2idx just for 
        for cat in categories_in_inst:
            if not(cat in self.total_cat2idx):
                self.total_cat2idx[cat] = len(self.total_cat2idx)
                # self.total_cat2idx[cat] = start_idx + cat_counter
                # cat_counter += 1

        # total_cat2idx["None"] = 1 + 1 + 5 * self.args.num_processes - 1  # self.args.num_processes == 1
        # if self.args.use_sem_policy:
        #     total_cat2idx["None"] = total_cat2idx["None"] + 23
        # self.total_cat2idx = total_cat2idx
        self.goal_idx2cat = {v: k for k, v in self.total_cat2idx.items()}
        print("self.goal_idx2cat is", self.goal_idx2cat)
        self.cat_list = categories_in_inst
        # self.args.num_sem_categories = 1 + 1 + 1 + 5 * self.args.num_processes
        # if self.args.use_sem_policy:
        #     self.args.num_sem_categories = self.args.num_sem_categories + 23
    
    def reset_map(self):
        # Calculating full and local map sizes
        map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w, self.local_h = (
            int(self.full_w / self.args.global_downscaling),
            int(self.full_h / self.args.global_downscaling)
        )
        # Initializing full and local map
        ### Full map consists of multiple channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations
        ### 5,6,7,.. : Semantic Categories
        nc = self.args.num_sem_categories + 4  # num channels
        self.full_map = torch.zeros(
            self.num_scenes, nc, self.full_w, self.full_h
        ).float().to(self.map_gpu)
        self.local_map = torch.zeros(
            self.num_scenes, nc, self.local_w, self.local_h
        ).float().to(self.map_gpu)
        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.map_gpu)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.map_gpu)
        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))
        # Local Map Boundaries
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)
        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
        # self.curr_loc = [0.,0.,0.]

        for e in range(self.num_scenes):
            # np.random.seed(e)  <-- we've already set the seeds
            c1 = np.random.choice(self.local_w)
            # np.random.seed(e + 1000)  <-- we've already set the seeds
            c2 = np.random.choice(self.local_h)
            if self.args.debug:
                c1, c2 = (120, 228)
            self.global_goals.append((c1, c2))
        
        #self.init_map_and_pose()
        #self.init_map_and_pose_for_env(0) # Reset map for 0 env

    def update_map(self, rgb: np.ndarray):
        # rgb is not used inside self.sem_map_module() but we keep it, since
        # self.sem_map_module() believes the first 3 channels are occupied by the rgb
        rgb = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)  # shape == (h, w, 3)
        obs = np.concatenate(
            (rgb, np.expand_dims(self.depth, 2), self.sem_seg_pred), axis=2
        )

        ds = self.args.env_frame_width // self.args.frame_width  # Downscaling factor
        if ds != 1:
            obs = obs[ds//2::ds, ds//2::ds]
            # rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            # depth = depth[ds//2::ds, ds//2::ds]
            # sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]
        obs = torch.tensor(obs.transpose(2, 0, 1)).unsqueeze(0).float().to(self.map_gpu)

        if abs(self.accumulated_pose.sum() - 0.) > 1e-3:  # LookUp/LookDown failed
            dx, dy, do = self.accumulated_pose
            self.info['sensor_pose'][0] += abs(dx + dy)
            self.info['sensor_pose'][2] += do
            self.accumulated_pose = np.array([0.0, 0.0, 0.0])
        poses = torch.from_numpy(
            np.asarray([self.info['sensor_pose']])
        ).float().to(self.map_gpu)
        # print('POSES: ', poses.cpu().numpy(), self.local_pose.cpu().numpy())
        self.sem_map_module.set_view_angles([self.camera_horizon])
        _, self.local_map, _, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose
        )

        locs = self.local_pose.cpu().numpy()
        # In FILM, planner_pose_inputs is not updated in the very beginning,
        # but after predicting the first action is updated. Since self.origins
        # is zero array at the beginning, we can "always" update planner_pose_inputs
        # and stay consistent with FILM
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        for e in range(self.num_scenes):
            if self.do_not_update_cat_s[e] is not None:
                cn = self.do_not_update_cat_s[e] + 4
                self.local_map[e, cn, :, :] = torch.zeros(self.local_map[0, 0, :, :].shape)

        for e in range(self.num_scenes):
            if (
                self.args.delete_from_map_after_move_until_visible
                and (self.move_until_visible_cycled or self.delete_lamp)
            ):
                # ep_num = args.from_idx + traj_number[e] * num_scenes + e  <-- is not used
                # Get the label that is closest to the current goal
                # cn = infos[e]['goal_cat_id'] + 4
                cn = self.goal_idx + 4
                start_x, start_y, _, gx1, gx2, gy1, gy2 = self.planner_pose_inputs[e]
                gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
                r, c = start_y, start_x
                start = [int(r * 100.0 / self.args.map_resolution - gx1),
                         int(c * 100.0 / self.args.map_resolution - gy1)]
                map_pred = np.rint(self.local_map[e, 0, :, :].cpu().numpy())
                assert self.local_map[e, 0, :, :].shape[0] == 240
                start = pu.threshold_poses(start, map_pred.shape)

                lm = self.local_map[e, cn, :, :].cpu().numpy()
                lm = (lm > 0).astype(int)
                lm = skimage.morphology.binary_dilation(lm, skimage.morphology.disk(4))
                lm = lm.astype(int)
                connected_regions = skimage.morphology.label(lm, connectivity=2)
                unique_labels = [i for i in range(0, np.max(connected_regions) + 1)]
                min_dist = 1000000000
                min_lab = -1
                for lab in unique_labels:
                    if lab != 0:  # <==> lab != background
                        wheres = np.where(connected_regions == lab)
                        center = int(np.mean(wheres[0])), int(np.mean(wheres[1]))
                        dist_pose = math.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)
                        min_dist = min(min_dist, dist_pose)
                        if min_dist == dist_pose:
                            min_lab = lab
                # Delete that label
                self.wheres_delete_s[e][np.where(connected_regions == min_lab)] = 1
            else:
                self.wheres_delete_s[e] = np.zeros((240, 240))
        for e in range(self.num_scenes):
            # cn = infos[e]['goal_cat_id'] + 4
            cn = self.goal_idx + 4
            wheres = np.where(self.wheres_delete_s[e])
            self.local_map[e, cn, :, :][wheres] = 0.0

    def set_back_to_angle(self, angle_arg):
        # look_failure = False
        view_angle_copy = copy.deepcopy(int(self.camera_horizon))
        if abs(view_angle_copy - angle_arg) > 5:
            if view_angle_copy > angle_arg:
                return 'LookUp_15'
                # Looking down like 60 degrees
                # Look up until 45
                # times_15 = int((view_angle_copy-angle_arg)/15)
                # for i in range(times_15):
                #     obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("LookUp_15")
                #     if self.args.look_no_repeat and not(success):
                #         look_failure = True
                #         break
                # if not(look_failure) and abs(int(self.camera_horizon) - angle_arg) > 5:
                #     angle = (view_angle_copy - angle_arg) - 15 * times_15
                #     obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("LookUp_" + str(angle))
            else:
                return 'LookDown_15'
                # times_15 = int((angle_arg-view_angle_copy)/15)
                # for i in range(times_15):
                #     obs, rew, done, info, success, _, target_instance, err, _ = \
                #                         self.va_interact_new("LookDown_15")
                    # print("Looked down once")
                    # if self.args.look_no_repeat and not(success):
                    #     look_failure = True
                    #     break
                # if not(look_failure) and abs(int(self.camera_horizon) - angle_arg) > 5:
                #     angle = (angle_arg - view_angle_copy) - 15 * times_15
                #     obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("LookDown_" + str(angle))

            # return obs, rew, done, info, success, _, target_instance, err, _
        else:
            # return self.va_interact_new("LookUp_0")
            return "LookUp_0"

    def plan_act(self) -> str:
        self.number_of_plan_act_calls += 1
        ################
        # From main.py #
        ################
        self.newly_goal_set = False
        # Moments for the goal prediction differ from FILM because LookUp_0s are taken 
        # into account. But this shouldn't affect the behaviour (and hence
        # the performance) much.
        if self.number_of_plan_act_calls % self.args.num_local_steps == 0:
            self.newly_goal_set = True
            self.get_neural_goal()

        planner_inputs = [{} for e in range(self.num_scenes)]

        self.goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]
        for e in range(self.num_scenes):
            self.goal_maps[e][self.global_goals[e][0], self.global_goals[e][1]] = 1

            # ep_num = args.from_idx + traj_number[e] * num_scenes + e  <-- is not used
            # cn = infos[e]['goal_cat_id'] + 4
            cn = self.goal_idx + 4
            # prev_cns[e] = cn  <-- is not used
            # cur_goal_sliced = next_step_dict_s[e]['current_goal_sliced']  <-- is not used

            if self.local_map[e, cn, :, :].sum() != 0.:
                # ep_num = args.from_idx + traj_number[e] * num_scenes + e  <-- is not used
                cat_semantic_map = self.local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map 
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                wheres = np.where(self.wheres_delete_s[e])
                cat_semantic_scores[wheres] = 0
                if np.sum(cat_semantic_scores) != 0:
                    self.goal_maps[e] = cat_semantic_scores
                
                if np.sum(cat_semantic_scores) != 0:
                    self.found_goal[e] = 1
                    self.goal_spotted_s[e] = True
                else:
                    if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
                        self.found_goal[e] = 0
                        self.goal_spotted_s[e] = False
            else:
                if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
                    self.found_goal[e] = 0
                    self.goal_spotted_s[e] = False

        for e, p_input in enumerate(planner_inputs):
            p_input['newly_goal_set'] = self.newly_goal_set
            p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = self.planner_pose_inputs[e]
            p_input['goal'] = self.goal_maps[e]
            p_input['found_goal'] = self.found_goal[e]
            p_input['list_of_actions_pointer'] = 0  # is not used
            p_input['goal_spotted'] = self.goal_spotted_s[e]

        #########################################################
        # From Sem_Exp_Env_Agent_Thor.plan_act_and_preprocess() #
        #########################################################
        action = None
        moved_until_visible = False
        side_stepped = None
        sdroate_direction = None
        self.opp_side_step = False

        traversible, cur_start, cur_start_o = self.get_traversible(planner_inputs[0])
        if self.side_step_order in [1, 2]:
            prev_side_step_order = copy.deepcopy(self.side_step_order)
            # obs, rew, done, info, success, target_instance, err = self.side_step(self.step_dir, cur_start_o, cur_start, traversible)
            action = self.side_step(self.step_dir, cur_start_o, cur_start, traversible)
            # obs, seg_print = self.preprocess_obs_success(success, obs)  <-- will be done after action execution
            # self.info = info  <-- we only need info['sensor_pose'] which will be updated
            # self.interaction_mask = None  <-- is useless here in FILM
            self.doing_side_step_order_1_2 = True
            self.side_step_helper_save = {
                'prev_side_step_order': prev_side_step_order,
                'goal_spotted': planner_inputs[0]['goal_spotted']
            }
            # The following code does not depend on execution results
            # and can be left here
            if self.side_step_order == 0:
                side_stepped = self.step_dir
            self.last_action_ogn = action  # FILM lacks this update

        elif self.lookDownUpLeft_count in range(1, 4):  # in 1, 2, 3
            if self.args.debug:
                print("Tried to lookdownupleft")
            if self.lookDownUpLeft_count == 1:
                # cur_hor = np.round(self.camera_horizon, 4)
                # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(0)
                self.cam_target_angle = 0
                self.is_changing_cam_angle = True
                action = self.set_back_to_angle(0)
                self.lookDownUpLeft_count += 1
            elif self.lookDownUpLeft_count == 2:
                # look down back to 45
                cur_hor = np.round(self.camera_horizon, 4)
                # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
                self.cam_target_angle = 45
                self.is_changing_cam_angle = True
                action = self.set_back_to_angle(45)
                self.lookDownUpLeft_count += 1
            elif self.lookDownUpLeft_count == 3:
                action = "RotateLeft_90"
                # obs, rew, done, info, success, _, target_instance, err, _ = \
                #             self.va_interact_new("RotateLeft_90")
                self.lookDownUpLeft_count = 0

            # obs, seg_print = self.preprocess_obs_success(success, obs)
            self.last_action_ogn = action
            # self.info = info  <-- we only need info['sensor_pose'] which will be updated
            self.execute_interaction = False
            self.interaction_mask = None
        elif self.execute_interaction:
            self.last_action_ogn = action = 'StopNav'
            self.execute_interaction = False
        else:
            if self.prev_number_action != 0:
                action = self._plan(planner_inputs[0])
                if action == 0:
                    self.prev_number_action = 0
                action_dict = {
                    0: "StopNav", 1: "MoveAhead_25", 2: "RotateLeft_90", 
                    3: "RotateRight_90", 4: "LookDown_90", 5: "LookDownUpLeft"
                }  # 4 is never predicted
                self.last_action_ogn = action = action_dict[action]

            # `repeated_rotation` is not used in FILM, so it is skipped

            if self.prev_number_action == 0:  # Stop outputted now or before
                if (
                    self.use_stop_analysis
                    and self.args.approx_error_message
                    and not self.last_not_nav_subtask_success
                    and getattr(self.last_not_nav_subtask, "action", "None") in ["OpenObject", "CloseObject"]
                ):
                    action = self.move_behind_correction(planner_inputs[0])

                # Runs anyway, so there is no check for self.use_stop_analysis
                elif (
                    not self.args.no_rotate_sidestep
                    and (
                        self.last_three_sidesteps[0] is not None
                        and self.prev_sidestep_success == False
                    )
                ):
                    # Rotate to the failed direction 
                    if self.args.debug:
                        print("Rotating because sidestepping failed")
                    self.update_loc(planner_inputs[0])
                    if self.last_three_sidesteps[0] == 'right':
                        sdroate_direction = "Right"
                    elif self.last_three_sidesteps[0] == 'left':
                        sdroate_direction = "Left"
                    action = 'Rotate' + sdroate_direction + '_90'
                    # obs, rew, done, info, success, _, target_instance, err, _ = self.va_interact_new("Rotate" +sdroate_direction+ "_90")
                    self.update_last_three_sidesteps("Rotate" + sdroate_direction)

                # Must be cautious pointers  <-- ???
                elif (
                    self.use_stop_analysis
                    and self.is_visible_from_mask(self.interaction_mask)
                ):
                    # Sidestep
                    wd = self.which_direction()
                    if self.args.debug:
                        print("wd is", wd)
                    if wd <= 100:
                        step_dir = 'left'
                        if self.args.debug:
                            print("sidestepping to left")
                    elif wd > 200:
                        step_dir = 'right'
                        if self.args.debug:
                            print("sidestepping to right")
                    else:
                        step_dir = None
                        if self.args.debug:
                            print("skipping sidestepping")
                    action = self.side_step_correction(step_dir, planner_inputs[0])

                # Not visible
                elif self.use_stop_analysis:
                    moved_until_visible = True
                    action = self.move_until_visible_correction(planner_inputs[0])

                # If the stop analysis is not allowed, stop the execution
                else:
                    action = 'StopNav'

            elif self.last_action_ogn == "LookDownUpLeft":
                cur_hor = np.round(self.camera_horizon, 4)
                if abs(cur_hor - 45) > 5:
                    # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
                    self.cam_target_angle = 45
                    self.is_changing_cam_angle = True
                    action = self.set_back_to_angle(45)
                else:
                    action = "LookUp_0"

                self.lookDownUpLeft_count = 1
            
            self.in_plan_act_else = True
            self.plan_act_else_save = {
                'goal_spotted': planner_inputs[0]['goal_spotted'],
                'opp_side_step': self.opp_side_step
            }
            self.last_action_ogn = action

        self.delete_lamp = \
            self.mvb_num_in_cycle != 0 and self.goal_name == 'FloorLamp' \
            and getattr(self.last_not_nav_subtask, 'action', 'None') == 'ToggleObjectOn'
        if self.args.no_delete_lamp:
            self.delete_lamp = False

        if not moved_until_visible:
            self.mvb_num_in_cycle = 0
            self.mvb_which_cycle = 0

        self.rotate_aftersidestep = sdroate_direction
        self.move_until_visible_cycled = \
            self.mvb_which_cycle != 0 and self.mvb_num_in_cycle == 0

        # Since FILM checks a sidestep success without involving the actions
        # success, we can use it before the action is actually done
        if self.side_step_order == 0 and side_stepped is None:
            self.prev_sidestep_success = True
            self.update_last_three_sidesteps(side_stepped)
        return action

    def plan_act_else_helper(self, rgb: np.ndarray):
        # obs, seg_print = self.preprocess_obs_success(success, obs)
        # if self.args.film_use_oracle_seg:
        #     self.interaction_mask = self.get_instance_mask_from_obj_type(self.goal_name)
        # else:
        #     print("obj type for mask is", self.goal_name)
        #     self.interaction_mask = self.sem_seg_get_instance_mask_from_obj_type(self.goal_name)
        self.interaction_mask = self.seg_model.get_interaction_mask(
            self.img_transform_seg(rgb), self.goal_name, check_zero_mask=False
        )
        
        visible = self.is_visible_from_mask(self.interaction_mask)

        # self.last_action_ogn = action  <-- is positioned in plan_act()
        # self.info = info  <-- we only need info['sensor_pose'] which will be updated
        
        # list_of_actions = planner_inputs['list_of_actions']  <-- is not used by us
        # pointer = planner_inputs['list_of_actions_pointer']  <-- is not used by FILM
        # interaction = list_of_actions[pointer][1]  <-- is not used by us

        # Meaningless code:
        # if self.args.stricter_visibility <= 1.5:
        #     if self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility) and self.whether_center():
        #         self.prev_number_action == 0

        # pointer = planner_inputs['list_of_actions_pointer']  <-- is replaced by self.cur_nav_subtask
        goal_spotted = self.plan_act_else_save['goal_spotted']
        opp_side_step = self.plan_act_else_save['opp_side_step']
        
        if self.cur_nav_subtask not in self.caution_pointers:
            self.execute_interaction = goal_spotted and visible
        else:  # caution pointers
            whether_center = self.whether_center()
            self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center
            if opp_side_step:
                self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0

    def move_behind_correction(self, planner_inputs: Optional[dict] = None) -> str:
        if planner_inputs is None:
            planner_inputs = {
                'pose_pred': self.planner_pose_inputs[0],
                'map_pred': self.local_map[0, 0, :, :].cpu().numpy()
            }

        self.update_loc(planner_inputs)
        self.is_moving_behind = True
        action = self.move_behind()
        if self.args.debug:
            # print("Moved behind!")
            print('Moving behind!')
        return action
    
    def right_side_step_correction(self) -> str:
        return self.side_step_correction(step_dir='right')

    def left_side_step_correction(self) -> str:
        return self.side_step_correction(step_dir='left')

    def side_step_correction(
        self, step_dir, planner_inputs: Optional[dict] = None
    ) -> str:
        if planner_inputs is None:
            planner_inputs = {
                'pose_pred': self.planner_pose_inputs[0],
                'map_pred': self.local_map[0, 0, :, :].cpu().numpy()
            }
        traversible, cur_start, cur_start_o = self.get_traversible(planner_inputs)

        self.update_loc(planner_inputs)
        if (
            not self.args.no_opp_sidestep
            and self.last_three_sidesteps == ['left', 'right', 'left']
            or self.last_three_sidesteps == ['right', 'left', 'right']
        ):
            self.opp_side_step = True
            if step_dir is None:
                opp_step_dir = None
                action = 'LookUp_0'
                # obs, rew, done, info, success, _, target_instance, err, _ = self.va_interact_new("LookUp_0") #pass
            else:
                if step_dir == 'left':
                    opp_step_dir = 'right'
                else:
                    opp_step_dir = 'left'
                action = self.side_step(opp_step_dir, cur_start_o, cur_start, traversible)
                # obs, rew, done, info, success, target_instance, err = self.side_step(opp_step_dir, cur_start_o, cur_start, traversible)
            side_stepped = opp_step_dir
        else:
            self.opp_side_step = False
            if step_dir is not None:
                action = self.side_step(step_dir, cur_start_o, cur_start, traversible)
                # obs, rew, done, info, success, target_instance, err = self.side_step(step_dir, cur_start_o, cur_start, traversible)
            else:
                action = 'LookUp_0'
                # obs, rew, done, info, success, _, target_instance, err, _=  self.va_interact_new("LookUp_0") #pass
            side_stepped = step_dir
        if self.args.debug:
            print("last three side stepped", self.last_three_sidesteps)

        # Since FILM checks a sidestep success without involving the actions
        # success, we can use it before the action is actually done
        if self.side_step_order == 0 and side_stepped is None:
            self.prev_sidestep_success = True
            self.update_last_three_sidesteps(side_stepped)
        return action

    def move_until_visible_correction(
        self, planner_inputs: Optional[dict] = None, rgb: Optional[np.ndarray] = None
    ) -> str:
        if planner_inputs is None:
            planner_inputs = {
                'pose_pred': self.planner_pose_inputs[0],
                'map_pred': self.local_map[0, 0, :, :].cpu().numpy()
            }

        # Check whether the object has become visible:
        if rgb is not None:
            self.interaction_mask = self.seg_model.get_interaction_mask(
                self.img_transform_seg(rgb), self.goal_name, check_zero_mask=False
            )
            visible = self.is_visible_from_mask(self.interaction_mask)
        else:
            visible = False
        goal_spotted = self.goal_spotted_s[0]
        opp_side_step = self.opp_side_step
        if self.cur_nav_subtask not in self.caution_pointers:
            self.execute_interaction = goal_spotted and visible
        else:  # caution pointers
            whether_center = self.whether_center()
            self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center
            if opp_side_step:
                self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0

        # Decide whether to continue:
        if self.execute_interaction:
            self.in_move_until_visible_correction = False
            return 'StopNav'
        else:
            # moved_until_visible = True
            if self.args.debug:
                print("moving until visible")
                print("current horizon is", self.camera_horizon)
                print("move until visible order is", self.move_until_visible_order)
            self.update_loc(planner_inputs)
            # obs, rew, done, info, success, target_instance, err  = self.move_until_visible()
            action = self.move_until_visible()
            self.mvb_num_in_cycle += 1
            if self.mvb_num_in_cycle == 12:
                print("Went through one cycle of move until visible, step num is", self.steps_taken)
                self.mvb_which_cycle += 1
                self.mvb_num_in_cycle = 0
                if self.args.delete_from_map_after_move_until_visible:
                    self.prev_number_action = 100  # Release "stop outputted"

            self.delete_lamp = \
                self.mvb_num_in_cycle != 0 and self.goal_name == 'FloorLamp' \
                and getattr(self.last_not_nav_subtask, 'action', 'None') == 'ToggleObjectOn'
            if self.args.no_delete_lamp:
                self.delete_lamp = False
            self.move_until_visible_cycled = \
                self.mvb_which_cycle != 0 and self.mvb_num_in_cycle == 0
            return action

    def get_neural_goal(self):
        for e in range(self.num_scenes):
            # if wait_env[e] == 1:  # New episode
            #     wait_env[e] = 0.

            self.full_map[
                e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]
            ] = self.local_map[e]
            self.full_pose[e] = self.local_pose[e] \
                + torch.from_numpy(self.origins[e]).to(self.depth_gpu).float()

            locs = self.full_pose[e].cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.lmb[e] = self.get_local_map_boundaries(
                (loc_r, loc_c), (self.local_w, self.local_h), 
                (self.full_w, self.full_h)
            )

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [
                self.lmb[e][2] * self.args.map_resolution / 100.0, 
                self.lmb[e][0] * self.args.map_resolution / 100.0,
                0.
            ]

            self.local_map[e] = self.full_map[
                e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
            ]
            self.local_pose[e] = self.full_pose[e] \
                - torch.from_numpy(self.origins[e]).to(self.depth_gpu).float()

        locs = self.local_pose.cpu().numpy()

        for e in range(self.num_scenes):
            # Just reconst the common map save objects
            map_reconst = torch.zeros((4 + len(self.map_save_large_objects), 240, 240))
            map_reconst[:4] = self.local_map[e][:4]
            # test_see = {}
            # map_reconst[4+self.map_save_large_objects2idx['SinkBasin']] = self.local_map[e][4+1]
            # test_see[1] = 'SinkBasin'

            # Relies on the keys order in self.total_cat2idx
            map_reconst[4 : 4 + len(self.map_save_large_objects)] = self.local_map[e][
                4 : 4 + len(self.map_save_large_objects)
            ]
            # start_idx = 2
            # for cat in self.map_save_large_objects2idx.keys():
                # if not (cat == 'SinkBasin'):
                    # map_reconst[4+self.map_save_large_objects2idx[cat]] = self.local_map[e][4+start_idx]
                    # test_see[start_idx] = cat
                    # start_idx += 1

            if self.local_map[e][0][120, 120] == 0:
                mask = np.zeros((240, 240))
                connected_regions = skimage.morphology.label(1-self.local_map[e][0].cpu().numpy(), connectivity=2)
                connected_lab = connected_regions[120, 120]
                mask[np.where(connected_regions == connected_lab)] = 1
                mask[np.where(skimage.morphology.binary_dilation(
                    self.local_map[e][0].cpu().numpy(), 
                    skimage.morphology.square(4)
                ))] = 1
            else:
                dilated = skimage.morphology.binary_dilation(
                    self.local_map[e][0].cpu().numpy(), skimage.morphology.square(4)
                )
                mask = skimage.morphology.convex_hull_image(dilated).astype(float)
            mask_grid = self.into_grid(torch.tensor(mask), 8).cpu()
            where_ones = len(torch.where(mask_grid)[0])
            mask_grid = mask_grid.repeat(73, 1).view(73, -1).numpy()

            if self.goal_name in self.all_objects2idx and self.steps_taken >= 30:
                pred_probs = self.Unet_model(map_reconst.unsqueeze(0).to(self.depth_gpu))
                pred_probs = pred_probs.view(73, -1)
                pred_probs = self.softmax(pred_probs).cpu().numpy()

                # args.explore_probs = 0., so we don't need this:
                # pred_probs = (1-args.explore_prob) * pred_probs + args.explore_prob * mask_grid * 1 / float(where_ones)

                # Now sample from pred_probs according to the goal idx
                if self.goal_name == 'FloorLamp':
                    pred_probs = (
                        pred_probs[self.all_objects2idx[self.goal_name]]
                        + pred_probs[self.all_objects2idx['DeskLamp']]
                    ) / 2.0
                elif self.goal_name == 'DeskLamp':
                    pred_probs = (
                        pred_probs[self.all_objects2idx[self.goal_name]]
                        + pred_probs[self.all_objects2idx['FloorLamp']]
                    ) / 2.0
                else:
                    pred_probs = pred_probs[self.all_objects2idx[self.goal_name]]

            else:
                pred_probs = mask_grid[0] * 1 / float(where_ones)

            # if args.explore_prob == 1.0:  <-- we don't need this, since args.explore_probs = 0.
            #     mask_wheres = np.where(mask.astype(float))
            #     np.random.seed(next_step_dict_s[e]['steps_taken'])
            #     s_i = np.random.choice(len(mask_wheres[0]))
            #     x_240, y_240 = mask_wheres[0][s_i], mask_wheres[1][s_i]
            # else:

            # Now sample one index
            # np.random.seed(next_step_dict_s[e]['steps_taken'])  <-- we've already set the seeds
            pred_probs = pred_probs.astype('float64')
            pred_probs = pred_probs.reshape(64)
            pred_probs = pred_probs / np.sum(pred_probs)

            chosen_cell = np.random.multinomial(1, pred_probs.tolist())
            chosen_cell = np.where(chosen_cell)[0][0]
            chosen_cell_x = int(chosen_cell/8)
            chosen_cell_y = chosen_cell % 8

            # Sample among this mask
            mask_new = np.zeros((240, 240))
            mask_new[
                chosen_cell_x*30:chosen_cell_x*30+30, 
                chosen_cell_y*30: chosen_cell_y*30 + 30
            ] = 1
            self.mask_new = mask_new = mask_new * mask
            if np.sum(mask_new) == 0:
                # np.random.seed(next_step_dict_s[e]['steps_taken'])  <-- we've already set the seeds
                chosen_i = np.random.choice(len(np.where(mask)[0]))
                x_240 = np.where(mask)[0][chosen_i]
                y_240 = np.where(mask)[1][chosen_i]
            else:
                # np.random.seed(next_step_dict_s[e]['steps_taken'])  <-- we've already set the seeds
                chosen_i = np.random.choice(len(np.where(mask_new)[0]))
                x_240 = np.where(mask_new)[0][chosen_i]
                y_240 = np.where(mask_new)[1][chosen_i]

            self.global_goals[e] = [x_240, y_240]
            # test_goals = np.zeros((240, 240))
            # test_goals[x_240, y_240] = 1

    @staticmethod
    def into_grid(ori_grid, grid_size):
        one_cell_size = math.ceil(240 / grid_size)
        return_grid = torch.zeros(grid_size, grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                if torch.sum(ori_grid[
                    one_cell_size * i : one_cell_size*(i + 1), 
                    one_cell_size * j : one_cell_size*(j + 1)
                ].bool().float()) > 0:
                    return_grid[i, j] = 1
        return return_grid
        
    def reset_nav(
        self, scene_id=None, episode_id=None, episode_name=None, goal_obj=None
    ) -> None:            
        # self.list_of_actions, self.categories_in_inst, self.second_object, self.caution_pointers = get_list_of_highlevel_actions(self.traj_data, args_nonsliced=False) 
        self.accumulated_pose = np.array([0.0, 0.0, 0.0])            
        self.info['sensor_pose'] = [0., 0., 0.]
        
        map_shape = (1200 // 5, 1200 // 5)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.col_width = 5
        self.curr_loc = [1200 / 100.0 / 2.0, 1200 / 100.0 / 2.0, 0.]
            
        self.o = 0.0
        self.o_behind = 0.0    
        # self.path_length = 1e-5
        # return self.event
   
    def get_approximate_success(self, rgb: np.ndarray) -> bool:
        wheres = np.where(self.prev_rgb != rgb)
        wheres_ar = np.zeros(self.prev_rgb.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)
        # if (self.last_action in ['OpenObject', 'CloseObject']) and max_area > 500:
        #     success = True
        if max_area > 100:
            success = True
        else:
            success = False
        return success

    """
    def get_pose_change_approx(self, last_action, whether_success):
        
        if not(whether_success):
            return 0.0, 0.0, 0.0
        else:
            if last_action==0:
                dx, dy, do = 0.25, 0.0, 0.0
            elif last_action==1:
                dx, dy = 0.0, 0.0
                do = np.pi/2
            elif last_action==2:
                dx, dy = 0.0, 0.0
                do = -np.pi/2
            else:
                dx, dy, do = 0.0, 0.0, 0.0
                
            if last_action==4:
                self.camera_horizon +=15
            elif last_action==5: 
                self.camera_horizon -=15

            return dx, dy, do 
    """
        
    def get_pose_change_approx_relative(self, last_action, whether_success):
        if not(whether_success):
            return 0.0, 0.0, 0.0
        else:
            if last_action == 'MoveAhead':
                do = 0.0
                if abs(self.o + 2*np.pi) <=1e-1 or abs(self.o) <=1e-1 or abs(self.o - 2*np.pi) <=1e-1: #o is 0
                    dx = 0.25
                    dy = 0.0
                elif abs(self.o + 2*np.pi - np.pi/2) <=1e-1 or abs(self.o - np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - np.pi/2) <=1e-1:
                    dx = 0.0
                    dy = 0.25
                elif abs(self.o + 2*np.pi - np.pi) <=1e-1 or abs(self.o - np.pi) <=1e-1 or abs(self.o - 2*np.pi - np.pi) <=1e-1:
                    dx = -0.25
                    dy = 0.0
                elif abs(self.o + 2*np.pi - 3*np.pi/2) <=1e-1 or abs(self.o - 3*np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - 3*np.pi/2) <=1e-1:
                    dx = 0.0
                    dy = -0.25
                else:
                    raise Exception("angle did not fall in anywhere")
            elif last_action == 'RotateLeft':
                dx, dy = 0.0, 0.0
                do = np.pi/2
            elif last_action == 'RotateRight':
                dx, dy = 0.0, 0.0
                do = -np.pi/2
            else:
                dx, dy, do = 0.0, 0.0, 0.0

        self.o = self.o + do
        if self.o >= np.pi - 1e-1:
            self.o -= 2 * np.pi
        # Added by us, since we got "angle did not fall in anywhere"
        elif self.o <= -(2 * np.pi - 1e-1):
            self.o += 2 * np.pi

        return dx, dy, do    

    # def get_instance_mask_seg_alfworld_both(self, rgb: np.ndarray):
    #     rgb = copy.deepcopy(rgb)

    #     ims = [rgb]
    #     im_tensors = [self.transform(i).to(device=self.depth_gpu) for i in ims]
    #     results_small = self.sem_seg_model_alfw_small(im_tensors)[0]

    #     im_tensors = [self.transform(i).to(device=self.depth_gpu) for i in ims]
    #     results_large = self.sem_seg_model_alfw_large(im_tensors)[0]

    #     desired_classes_small = []
    #     desired_classes_large = []

    #     desired_goal_small = []
    #     desired_goal_large = []
    #     for cat_name in self.total_cat2idx:
    #         if not(cat_name in ["None", "fake"]):
    #             if cat_name in self.large:
    #                 large_class = self.large_objects2idx[cat_name]
    #                 desired_classes_large.append(large_class)
    #                 if cat_name == self.goal_name or (cat_name in self.cat_equate_dict and self.cat_equate_dict[cat_name] == self.goal_name):
    #                     desired_goal_large.append(large_class)
    #             elif cat_name in self.small:
    #                 small_class = self.small_objects2idx[cat_name]
    #                 desired_classes_small.append(small_class)
    #                 if cat_name == self.goal_name or (cat_name in self.cat_equate_dict and self.cat_equate_dict[cat_name] == self.goal_name):
    #                     desired_goal_small.append(small_class)
    #             else:
    #                 pass

    #     desired_goal_small = list(set(desired_goal_small))
    #     desired_goal_large = list(set(desired_goal_large))

    #     # FROM here
    #     indices_small = []
    #     indices_large = []

    #     for k in range(len(results_small['labels'])):
    #         if (
    #             results_small['labels'][k].item() in desired_classes_small
    #             and results_small['scores'][k] > self.args.sem_seg_threshold_small
    #         ):
    #             indices_small.append(k)
    #     for k in range(len(results_large['labels'])):
    #         if (
    #             results_large['labels'][k].item() in desired_classes_large
    #             and results_large['scores'][k] > self.args.sem_seg_threshold_large
    #         ):
    #             indices_large.append(k)

    #     # Done until here
    #     pred_boxes_small = results_small['boxes'][indices_small].detach().cpu()
    #     pred_classes_small = results_small['labels'][indices_small].detach().cpu()
    #     pred_masks_small = results_small['masks'][indices_small].squeeze(1).detach().cpu().numpy()  # pred_masks[i] has shape (300,300)
    #     if self.args.with_mask_above_05:
    #         pred_masks_small = (pred_masks_small > 0.5).astype(float)
    #     pred_scores_small = results_small['scores'][indices_small].detach().cpu()

    #     for ci in range(len(pred_classes_small)):
    #         if self.small_idx2small_object[int(pred_classes_small[ci].item())] in self.cat_equate_dict:
    #             cat = self.small_idx2small_object[int(pred_classes_small[ci].item())]
    #             pred_classes_small[ci] = self.small_objects2idx[self.cat_equate_dict[cat]]

    #     pred_boxes_large = results_large['boxes'][indices_large].detach().cpu()
    #     pred_classes_large = results_large['labels'][indices_large].detach().cpu()
    #     pred_masks_large = results_large['masks'][indices_large].squeeze(1).detach().cpu().numpy()  # pred_masks[i] has shape (300,300)
    #     if self.args.with_mask_above_05:
    #         pred_masks_large = (pred_masks_large > 0.5).astype(float)
    #     pred_scores_large = results_large['scores'][indices_large].detach().cpu()

	# 	# Make the above into a dictionary
    #     self.segmented_dict = {
    #         'small': {
    #             'boxes': pred_boxes_small, 
    #             'classes': pred_classes_small, 
    #             'masks': pred_masks_small, 
    #             'scores': pred_scores_small
    #             }, 
    #         'large': {
    #             'boxes': pred_boxes_large, 
    #             'classes': pred_classes_large, 
    #             'masks': pred_masks_large, 
    #             'scores': pred_scores_large
    #             }
    #     }

    # def segmentation_for_map(self):
    #     small_len = len(self.segmented_dict['small']['scores'])
    #     large_len = len(self.segmented_dict['large']['scores'])

    #     semantic_seg = np.zeros((self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
    #     for i in range(small_len):
    #         category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
    #         v = self.segmented_dict['small']['masks'][i]

    #         if not(category in self.total_cat2idx):
    #             pass
    #         else:
    #             cat = self.total_cat2idx[category]
    #             semantic_seg[:, :, cat] +=  v.astype('float')

    #     for i in range(large_len):
    #         category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
    #         v = self.segmented_dict['large']['masks'][i]

    #         if not(category in self.total_cat2idx):
    #             pass
    #         else:
    #             cat = self.total_cat2idx[category]
    #             semantic_seg[:, :, cat] += v.astype('float')

    #     return semantic_seg

    def get_sem_pred(self, rgb):
        # if self.args.film_use_oracle_seg:
        #     semantic_pred = self.segmentation_ground_truth()  # (300, 300, num_cat)
        # else:
        # 1. Get the instance segmentation
        self.seg_model(self.img_transform_seg(rgb))
        # self.get_instance_mask_seg_alfworld_both(rgb)

        # 2. Delete the target class from the segmentation 
        # if the object has already been placed:
        if (
            self.last_not_nav_subtask is not None
            and self.last_not_nav_subtask.action == 'PutObject'
            and self.last_not_nav_subtask_success
            and self.is_pick_two_objs_and_place_task
        ):
            self.seg_model.ignore_objects_on_recep(
                self.obj_class_in_pick_two_task, self.recep_class_in_pick_two_task
            )
            # if (self.agent.pointer==2  or (self.agent.pointer==1 and self.agent.last_success and self.agent.last_action_ogn == "PutObject"))and self.agent.task_type == 'pick_two_obj_and_place':
                # small_len = len(self.segmented_dict['small']['scores'])
                # large_len = len(self.segmented_dict['large']['scores'])

                # wheres = []
                # for i in range(large_len):
                #     category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
                #     v = self.segmented_dict['large']['boxes'][i]
                #     if category == self.recep_class_in_pick_two_task:
                #         wheres.append(v)

                # avoid_idx = []
                # for i in range(small_len):
                #     category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
                #     v = self.segmented_dict['small']['boxes'][i]
                #     if category == self.obj_class_in_pick_two_task:
                #         for where in wheres:
                #             if (
                #                 v[0] >= where[0] and v[1] >= where[1]
                #                 and v[2] <= where[2] and v[3] <= where[3]
                #             ):
                #                 avoid_idx.append(i)
                # avoid_idx = list(set(avoid_idx))

                # incl_idx = [i for i in range(small_len) if not(i in avoid_idx)]
                # self.segmented_dict['small']['boxes'] = self.segmented_dict['small']['boxes'][incl_idx]
                # self.segmented_dict['small']['classes'] = self.segmented_dict['small']['classes'][incl_idx]
                # self.segmented_dict['small']['masks'] = self.segmented_dict['small']['masks'][incl_idx]
                # self.segmented_dict['small']['scores'] = self.segmented_dict['small']['scores'][incl_idx]

        # 3. Get segmentation for the map making
        semantic_seg_pred = self.seg_model.get_semantic_seg()
        # semantic_pred = self.segmentation_for_map()

            # #3. visualize (get sem_vis)
            # if self.args.visualize or self.args.save_pictures:
            #     sem_vis = self.visualize_sem()
        return semantic_seg_pred

    # def sem_seg_get_instance_mask_from_obj_type_largest_only(self, object_type):
    #     mask = np.zeros((300, 300))
    #     small_len = len(self.segmented_dict['small']['scores'])
    #     large_len = len(self.segmented_dict['large']['scores'])
    #     max_area = -1

    #     if object_type in self.cat_equate_dict:
    #         object_type = self.cat_equate_dict[object_type]

    #     if object_type in self.large_objects2idx:
    #         # Get the highest score
    #         for i in range(large_len):
    #             category = self.large_idx2large_object[
    #                 self.segmented_dict['large']['classes'][i].item()
    #             ]
    #             if category == object_type:
    #                 v = self.segmented_dict['large']['masks'][i]
    #                 score = self.segmented_dict['large']['scores'][i]
    #                 area = np.sum(self.segmented_dict['large']['masks'][i])
    #                 max_area = max(area, max_area)
    #                 if max_area == area:    
    #                     mask = v.astype('float')
    #     else:
    #         for i in range(small_len):
    #             category = self.small_idx2small_object[
    #                 self.segmented_dict['small']['classes'][i].item()
    #             ]
    #             if category == object_type:
    #                 v = self.segmented_dict['small']['masks'][i]
    #                 score = self.segmented_dict['small']['scores'][i]
    #                 area = np.sum(self.segmented_dict['small']['masks'][i])
    #                 max_area = max(area, max_area)
    #                 if max_area == area:   
    #                     mask = v.astype('float')

    #     if np.sum(mask) == 0:
    #         mask = None
    #     return mask

    # def sem_seg_get_instance_mask_from_obj_type(self, object_type):
    #     mask = np.zeros((300, 300))
    #     small_len = len(self.segmented_dict['small']['scores'])
    #     large_len = len(self.segmented_dict['large']['scores'])
    #     max_score = -1

    #     if object_type in self.cat_equate_dict:
    #         object_type = self.cat_equate_dict[object_type]

    #     if object_type in self.large_objects2idx:
    #         #Get the highest score
    #         for i in range(large_len):
    #             category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
    #             if category == object_type:
    #                 v = self.segmented_dict['large']['masks'][i]
    #                 score = self.segmented_dict['large']['scores'][i]
    #                 max_score = max(score, max_score)
    #                 if score == max_score:    
    #                     mask =  v.astype('float')
    #     else:
    #         for i in range(small_len):
    #             category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
    #             if category == object_type:
    #                 v = self.segmented_dict['small']['masks'][i]
    #                 score = self.segmented_dict['small']['scores'][i]
    #                 max_score = max(score, max_score)
    #                 if score == max_score:
    #                     mask =  v.astype('float')

    #     if np.sum(mask) == 0:
    #         mask = None
    #     return mask

    # def get_instance_mask_from_obj_type_largest(self, object_type):
    #     mask = np.zeros((300, 300))
    #     max_area = 0
    #     for k, v in self.env.last_event.instance_masks.items():
    #         category = k.split('|')[0]
    #         category_last = k.split('|')[-1]
    #         if category in self.cat_equate_dict:
    #             category = self.cat_equate_dict[category]
    #         if 'Sliced' in category_last:
    #             category = category + 'Sliced'
    #         if 'Sink' in category and 'SinkBasin' in category_last:
    #             category =  'SinkBasin' 
    #         if 'Bathtub' in category and 'BathtubBasin' in category_last:
    #             category =  'BathtubBasin'
    #         if category == object_type:
    #             if np.sum(v) >= max_area:
    #                 max_area = np.sum(v)
    #                 mask = v
    #     if np.sum(mask) == 0:
    #         mask = None
    #     return mask

    # def get_instance_mask_from_obj_type(self, object_type):
    #     mask = np.zeros((300, 300))
    #     for k, v in self.env.last_event.instance_masks.items():
    #         category = k.split('|')[0]
    #         category_last = k.split('|')[-1]
    #         if category in self.cat_equate_dict:
    #             category = self.cat_equate_dict[category]
    #         if 'Sliced' in category_last:
    #             category = category + 'Sliced'
    #         if 'Sink' in category and 'SinkBasin' in category_last:
    #             category =  'SinkBasin' 
    #         if 'Bathtub' in category and 'BathtubBasin' in category_last:
    #             category =  'BathtubBasin'
    #         if category == object_type:
    #             mask = v
    #     if np.sum(mask) == 0:
    #         mask = None
    #     return mask
    
    # def segmentation_ground_truth(self):
    #     instance_mask = self.env.last_event.instance_masks
    #     semantic_seg = np.zeros((self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
        
    #     if self.args.ignore_categories:
    #         for k, v in instance_mask.items():
    #             category = k.split('|')[0]
    #             category_last = k.split('|')[-1]
    #             if 'Sliced' in category_last:
    #                 category = category + 'Sliced'

    #             if 'Sink' in category and 'SinkBasin' in category_last:
    #                 category =  'SinkBasin' 
    #             if 'Bathtub' in category and 'BathtubBasin' in category_last:
    #                 category =  'BathtubBasin'
                
    #             if not(category in self.total_cat2idx):
    #                 pass
    #             else:
    #                 cat = self.total_cat2idx[category]
    #                 semantic_seg[:, :, cat] = v.astype('float')
    #     else:
    #         for k, v in instance_mask.items():
    #             category = k.split('|')[0]
    #             try:
    #                 cat = self.total_cat2idx[category]
    #                 success = 1
    #             except:
    #                 success = 0
                
    #             if success == 1:
    #                 semantic_seg[:, :, cat] = v.astype('float')
    #             else:
    #                 pass
    #     return semantic_seg.astype('uint8')

    def depth_pred_later(self, sem_seg_pred, rgb: np.ndarray):
        rgb = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)  # shape (h, w, 3)
        rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255.

        if abs(self.camera_horizon - 0) < 5:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float())
        else:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())

        if abs(self.camera_horizon - 0) < 5:
            include_mask_prop = 1.  # self.args.valts_trustworthy_obj_prop0
        else:
            include_mask_prop = 1.  # self.args.valts_trustworthy_obj_prop
            
        depth_img = pred_depth.get_trustworthy_depth(
            max_conf_int_width_prop=0.9, include_mask=sem_seg_pred,
            include_mask_prop=include_mask_prop
        )  # default is 1.0
        depth_img = depth_img.squeeze().detach().cpu().numpy()
        self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
        self.learned_depth_frame = self.learned_depth_frame.reshape((50, 300, 300))
        self.learned_depth_frame = 5 * 1/50 * np.argmax(self.learned_depth_frame, axis=0)  # Now shape is (300, 300)
        del pred_depth
        # depth = depth_img

        depth_img = np.expand_dims(depth_img, 2)
        return depth_img
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1  # shape is (h, w)

        if self.picked_up:
           mask_err_below = depth < 0.5
           if not(self.picked_up_mask is None):
               mask_picked_up = self.picked_up_mask == 1
               depth[mask_picked_up] = 100.0
        else:
            mask_err_below = depth <0.0
        depth[mask_err_below] = 100.0

        depth = depth * 100
        return depth
    
    def is_visible_from_mask(self, mask, stricer_visibility_dist=1.5):
        # if not(self.args.learned_visibility):
        #     if mask is None or np.sum(mask) == 0:
        #         return False
        #     min_depth = np.min(self.event.depth_frame[np.where(mask)])
        #     return min_depth <= stricer_visibility_dist * 1000

        # else:
        if mask is None or np.sum(mask) == 0:
            return False
        min_depth = np.min(self.learned_depth_frame[np.where(mask)])
        return min_depth <= stricer_visibility_dist
    
    def update_loc(self, planner_inputs):
        self.last_loc = self.curr_loc

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2) 

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
            
        r, c = start_y, start_x
        start = [int(r * 100.0/self.args.map_resolution - gx1),
                 int(c * 100.0/self.args.map_resolution - gy1)]
        map_pred = np.rint(planner_inputs['map_pred'])
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0]-0:start[0]+1,
                                       start[1]-0:start[1]+1] = 1

    def update_last_three_sidesteps(self, new_sidestep):
        self.last_three_sidesteps = self.last_three_sidesteps[:2]
        self.last_three_sidesteps = [new_sidestep] + self.last_three_sidesteps
    
    def move_behind(self, cur_start_o=None, cur_start=None, traversible=None):
        self.consecutive_steps = True
        # self.final_sidestep = False  <-- is not needed since it's satisfied
        self.move_behind_progress += 1
        if self.move_behind_progress < 3:
            # obs, rew, done, info, success1, _, target_instance, err, _ = \
            #         self.va_interact_new("RotateLeft_90")
            # obs, rew, done, info, success2, _, target_instance, err, _ = \
            #         self.va_interact_new("RotateLeft_90")
            return 'RotateLeft_90'
        if self.move_behind_progress == 3:
            # obs, rew, done, info, success3, _, target_instance, err, _ = \
            #         self.va_interact_new("MoveAhead_25")
            return 'MoveAhead_25'
        if self.move_behind_progress < 6:
            # obs, rew, done, info, success4, _, target_instance, err, _ = \
            #     self.va_interact_new("RotateLeft_90")
            # self.final_sidestep = True
            # obs, rew, done, info, success5, _, target_instance, err, _ = \
            #         self.va_interact_new("RotateLeft_90")
            if self.move_behind_progress == 5:
                self.is_moving_behind = False
                self.move_behind_progress = 0
                self.final_sidestep = True
            return 'RotateLeft_90'
        assert False, f'{self.move_behind_progress} must be < 6'
        # self.consecutive_steps = False
        # self.final_sidestep = False
        # success = success1 and success2 and success3 and success4 and success5
        # return obs, rew, done, info, success, target_instance, err

    def move_until_visible_helper(self):
        self.move_until_visible_helper_progress += 1
        if self.move_until_visible_helper_progress == 1:
            return "RotateLeft_90"
            # obs, rew, done, info, success1, _, target_instance, err, _ = \
            # self.va_interact_new(action)
        if self.move_until_visible_helper_progress == 2:
            self.cam_target_angle = 0
            self.is_changing_cam_angle = True
            action = self.set_back_to_angle(0)
            # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(0)
            self.final_sidestep = True  # It's okay to set it here to True since LookX doesn't change the pose
            # obs, rew, done, info, success, _, target_instance, err, _ = \
            #         self.va_interact_new("LookUp_0")
            self.move_until_visible_helper_progress = 0
            self.in_move_until_visible_helper = False
            return action
        assert False, f'{self.move_until_visible_helper_progress} must be < 2'

    def move_until_visible(self):
        if (
            self.rotate_aftersidestep is not None
            and abs(int(self.camera_horizon) - 0) < 5
        ):  # Rotated after sidestep but no longer visible
            self.cam_target_angle = 45
            self.is_changing_cam_angle = True
            action = self.set_back_to_angle(45)
            # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
        else:
            self.move_until_visible_order = self.move_until_visible_order % 8
            order = self.move_until_visible_order
            if abs(int(self.camera_horizon) - 45) < 5: 
                if order <= 3:  # 0, 1, 2, 3
                    action = "RotateLeft_90"
                    # obs, rew, done, info, success1, _, target_instance, err, _ = \
                    # self.va_interact_new(action)
                elif order > 3:  # 4, 5, 6, 7
                    self.consecutive_steps = True
                    self.in_move_until_visible_helper = True
                    action = self.move_until_visible_helper()
                    # self.consecutive_steps = False
                    # self.final_sidestep = False
                self.move_until_visible_order += 1
            else:
                self.cam_target_angle = 45
                self.is_changing_cam_angle = True
                action = self.set_back_to_angle(45)
                # obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
        # try:
        #     success = success1
        # except:
        #     print("move until visible broken!")
        return action

    # Turn towards 
    def side_step(self, step_dir, cur_start_o, cur_start, traversible):
        print("side step called, order:", self.side_step_order)
        if self.side_step_order == 0:
            self.step_dir = step_dir
            if step_dir == 'left':
                action = 'RotateLeft_90'
                # Turn left, moveforward, turn right
                # obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("RotateLeft_90")
            elif step_dir == 'right':
                action = 'RotateRight_90'
                # obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("RotateRight_90")
            else:
                raise Exception("Invalid sidestep direction")
            self.side_step_order = 1
            self.update_last_three_sidesteps(step_dir)
        elif self.side_step_order == 2:
            if step_dir == 'right':
                action = 'RotateLeft_90'
                    # Turn left, moveforward, turn right
                    # obs, rew, done, info, success, _, target_instance, err, _ = \
                    #     self.va_interact_new("RotateLeft_90")
            elif step_dir == 'left':
                action = 'RotateRight_90'
                    # obs, rew, done, info, success, _, target_instance, err, _ = \
                    #     self.va_interact_new("RotateRight_90")
            else:
                print("exception raised")
                raise Exception("Invalid sidestep direction")
            self.side_step_order = 0
        # Move ahead
        elif self.side_step_order == 1:
            if hasattr(self.args, 'check_before_sidestep') and self.args.check_before_sidestep:
                print("checking consec moving!")
                xy = CH._which_direction(cur_start_o)
                whether_move = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, cur_start,  xy)
            else:
                whether_move = True
            if not whether_move:
                #self.print_log("not move because no space!")
                action = 'LookUp_0'
                # obs, rew, done, info, success, _, target_instance, err, _ = \
                #         self.va_interact_new("LookDown_0")
                success = False
                if not success:
                    print("side step check prevented from move!")
                self.prev_sidestep_success = success
            else:
                action = 'MoveAhead_25'
                # obs, rew, done, info, success, _, target_instance, err, _ = \
                #     self.va_interact_new("MoveAhead_25")
                self.prev_sidestep_success_needs_update = True
                # if not success:  <-- is in _side_step_order_1_helper()
                #     print("side step moved and failed!")
            self.side_step_order = 2
            # self.prev_sidestep_success = success  <-- is in _side_step_order_1_helper()
        else:
            print("exception raised")
            raise Exception("Invalid sidestep order")
        return action
    
    def _side_step_order_1_helper(self, whether_success):
        if not whether_success:
            print("side step moved and failed!")
        self.prev_sidestep_success = whether_success

    def side_step_helper(self, rgb: np.ndarray):
        # if prev_side_step_order == 2:  <-- is united with the second `if prev_side_step_order == 2`
        #     opp_side_step = self.opp_side_step
        # if self.args.film_use_oracle_seg:
        #     self.interaction_mask = self.get_instance_mask_from_obj_type(self.goal_name)
        # else:
        #     print("obj type for mask is", self.goal_name)
        #     self.interaction_mask = self.sem_seg_get_instance_mask_from_obj_type(self.goal_name)
        self.interaction_mask = self.seg_model.get_interaction_mask(
            self.img_transform_seg(rgb), self.goal_name, check_zero_mask=False
        )

        prev_side_step_order = self.side_step_helper_save['prev_side_step_order']
        goal_spotted = self.side_step_helper_save['goal_spotted']
        if prev_side_step_order == 2:
            opp_side_step = self.opp_side_step
            # pointer = planner_inputs['list_of_actions_pointer']  <-- is replaced with self.cur_nav_subtask
            visible = self.is_visible_from_mask(self.interaction_mask)
            if self.cur_nav_subtask not in self.caution_pointers:
                self.execute_interaction = goal_spotted and visible
            else:
                whether_center = self.whether_center()
                self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center
                if opp_side_step:
                    self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0
        else:
            self.execute_interaction = False

    def which_direction(self):
        if self.interaction_mask is None:
            return 150
        widths = np.where(self.interaction_mask != 0)[1] 
        center = np.mean(widths)
        return center

    def whether_center(self):
        if self.interaction_mask is None:
            return False
        wd = self.which_direction()
        if np.sum(self.interaction_mask) == 0:  # if target_instance not in frame
            return False
        elif wd >= 65 and wd<=235:
            return True
        else:
            return False

    def get_traversible(self, planner_inputs):
        # args = self.args
        map_pred = np.rint(planner_inputs['map_pred'])
        grid = map_pred

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r * 100.0/self.args.map_resolution - gx1),
                 int(c * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        # def delete_boundary(mat):
        #     new_mat = copy.deepcopy(mat)
        #     return new_mat[1:-1,1:-1]

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[x1:x2, y1:y2],
                    self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is", self.steps_taken)

        traversible = add_boundary(traversible)

        # obstacle dilation
        traversible = 1 - traversible
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True

        traversible = traversible * 1.

        return traversible, start, start_o
    
    def _plan(self, planner_inputs):
        if planner_inputs['newly_goal_set']:
            self.action_5_count = 0

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0/self.args.map_resolution - gx1),
                 int(c * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0]-0:start[0]+1,
                                       start[1]-0:start[1]+1] = 1

        xy = CH._which_direction(start_o)
        xy = np.array(xy)

        # Update collision map
        if self.last_action_ogn == "MoveAhead_25":
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 5
                self.col_width = min(self.col_width, 15)
            else:
                self.col_width = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if self.last_action_ogn == "MoveAhead_25":
                col_threshold = self.args.collision_threshold
            elif "LookUp" in self.last_action_ogn:
                col_threshold = self.args.collision_threshold

            if dist < col_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(round(r*100/self.args.map_resolution)), \
                               int(round(c*100/self.args.map_resolution))
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collision_map.shape)
                        self.collision_map[r,c] = 1
        
        step = 0.  # Is not used in _get_stg

        stg, stop, whether_real_goal = self._get_stg(
            map_pred, start, np.copy(goal), planning_window,
            planner_inputs['found_goal'], xy.tolist(), step,
            planner_inputs['exp_pred'], planner_inputs['goal_spotted'],
            planner_inputs['newly_goal_set'],
            planner_inputs['list_of_actions_pointer']
        )  # planner_inputs['list_of_actions_pointer'] is not used inside

        # Deterministic Local Policy
        if stop and whether_real_goal:
            action = 0
        elif stop:
            if self.action_5_count < 4:
                action = 5  # lookdown, lookup, left
                self.action_5_count += 1
            else:
                action = 2
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal)%360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > 45:
                action = 3 # Right
            elif relative_angle < -45:
                action = 2 # Left
            else:
                action = 1        

        return action

    def _get_stg(
        self,grid, start, goal, planning_window, found_goal, xy_forward, step,
        explored, goal_found, newly_goal_set, pointer
    ):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]

        if goal.shape == (240, 240):
            goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[x1:x2, y1:y2],
                    self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        traversible = add_boundary(traversible)

        # obstacle dilation
        traversible = 1 - traversible
        selem = skimage.morphology.disk(1)#change to 5?
        #selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True


        traversible = traversible * 1.

        goal_shape = goal.shape
        if newly_goal_set:
            if self.args.debug:
                print("newly goal set")
            self.prev_wall_goal = None
            self.dilation_deg = 0

        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            if self.args.debug:
                print("Center done")
            goal, centers = CH._get_center_goal(goal, pointer)  # pointer is not used inside

        goal_copy = copy.deepcopy(goal)
        goal_to_save = copy.deepcopy(goal)   

        planner = FMMPlanner(traversible, self.args, step_size=self.args.step_size)
        planner.save_t = self.steps_taken

        if not(self.prev_wall_goal is None) and not(goal_found):
            if self.args.debug:
                print("wall goal")
            goal = self.prev_wall_goal
            self.goal_visualize = delete_boundary(goal)

        if self.dilation_deg!=0: 
            if self.args.debug:
                print("dilation added")
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)

        if self.prev_wall_goal is None and self.dilation_deg==0:
            if self.args.debug:
                print("None of that")
            else:
                pass

        if goal_found:
            if self.args.debug:
                print("goal found!")
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                w_goal = np.random.choice(240)
                # np.random.seed(self.steps_taken + 1000)  <-- we've already set the seeds
                h_goal = np.random.choice(240)
                goal[w_goal, h_goal] = 1

        try:
            planner.set_multi_goal(goal)
        except:
            # Just set a random place as goal
            # np.random.seed(self.steps_taken)  <-- we've already set the seeds
            w_goal = np.random.choice(240)
            # np.random.seed(self.steps_taken + 1000)  <-- we've already set the seeds
            h_goal = np.random.choice(240)
            goal[w_goal, h_goal] = 1
            planner.set_multi_goal(goal)

        planner_broken, where_connected = CH._planner_broken(
            planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited
        )
        if self.args.debug:
            print("planner broken is", planner_broken)

        d_threshold = 60
        cur_wall_goal = False
        # If the goal is unattainable, dilate it until it becomes attainable
        # or choose a random goal among explored area
        if planner_broken and self.number_of_plan_act_calls > 0:
            if self.args.debug:
                print("Planner broken!")

            # goal in obstruction case 
            if goal_found or self.args.use_sem_policy:
                if self.args.debug:
                    if not(goal_found):
                        print("Goal in obstruction")
                    else:
                        print("Goal found, goal in obstruction")      
                    print(
                        "Really broken?", 
                        CH._planner_broken(
                            planner.fmm_dist, goal, traversible, start, 
                            self.steps_taken, self.visited
                        )[0]
                    )
                while (
                    CH._planner_broken(
                        planner.fmm_dist, goal, traversible, start, 
                        self.steps_taken, self.visited
                    )[0]
                    and (self.dilation_deg < d_threshold)
                ):
                    if self.prev_wall_goal is not None:
                        goal = copy.deepcopy(self.prev_wall_goal)
                    else:
                        goal = copy.deepcopy(goal_copy)
                    self.dilation_deg += 1
                    goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
                    if goal_found:  # add original goal area
                        goal = CH._block_goal(centers,  goal, original_goal, goal_found)
                        if np.sum(goal) == 0:
                            goal = goal_copy
                    try:
                        planner.set_multi_goal(goal)
                    except:
                        # Just set a random place as goal
                        # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                        w_goal = np.random.choice(240)
                        # np.random.seed(self.steps_taken + 1000)  <-- we've already set the seeds
                        h_goal = np.random.choice(240)
                        goal[w_goal, h_goal] = 1
                        planner.set_multi_goal(goal)
                    if self.args.debug:
                        print("dilation is", self.dilation_deg)
                        print(
                            "Sanity check passed in loop is",
                            CH._planner_broken(
                                planner.fmm_dist, goal, traversible, start,
                                self.steps_taken, self.visited
                            )[0]
                        )

                if self.dilation_deg == d_threshold:
                    if self.args.debug:
                        print("Switched to goal in wall after dilation > 45")
                    self.dilation_deg = 0

                    # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                    random_goal_idx = np.random.choice(len(where_connected[0]))  # Choose a random goal among explored area 
                    random_goal_ij = (where_connected[0][random_goal_idx], where_connected[1][random_goal_idx])
                    goal = np.zeros(goal_shape)
                    goal[random_goal_ij[0], random_goal_ij[1]] = 1
                    self.prev_wall_goal = goal
                    self.goal_visualize = delete_boundary(goal)
                    try:
                        planner.set_multi_goal(goal)
                    except:
                        # Just set a random place as goal
                        # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                        w_goal = np.random.choice(240)
                        # np.random.seed(self.steps_taken + 1000)  <-- we've already set the seeds
                        h_goal = np.random.choice(240)
                        goal[w_goal, h_goal] = 1
                        planner.set_multi_goal(goal)
                    cur_wall_goal = True
            else:
                if self.args.debug:
                    print("Goal in wall, or goal in obstruction although goal not found")
                # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                random_goal_idx = np.random.choice(len(where_connected[0]))  # Choose a random goal among explored area 
                random_goal_ij = (where_connected[0][random_goal_idx], where_connected[1][random_goal_idx])
                goal = np.zeros(goal_shape)
                goal[random_goal_ij[0], random_goal_ij[1]] = 1
                self.prev_wall_goal = goal
                self.goal_visualize = delete_boundary(goal)
                try:
                    planner.set_multi_goal(goal)
                except:
                    # Just set a random place as goal
                    # np.random.seed(self.steps_taken)  <-- we've already set the seeds
                    w_goal = np.random.choice(240)
                    # np.random.seed(self.steps_taken + 1000)  <-- we've already set the seeds
                    h_goal = np.random.choice(240)
                    goal[w_goal, h_goal] = 1
                    planner.set_multi_goal(goal)
                cur_wall_goal = True

            if self.args.debug:
                print(
                    "Sanity check passed is",
                    CH._planner_broken(
                        planner.fmm_dist, goal, traversible, start, 
                        self.steps_taken, self.visited
                    )[0]
                )

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state, found_goal = found_goal, decrease_stop_cond=decrease_stop_cond)
        self.fmm_dist = planner.fmm_dist

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        self.closest_goal = CH._get_closest_goal(start, goal)


        whether_real_goal = found_goal and not(cur_wall_goal)

        self.prev_goal_cos = copy.deepcopy(goal_to_save)
        self.prev_step_goal_cos = copy.deepcopy(goal)

        return (stg_x, stg_y), stop, whether_real_goal
    
    # def init_map_and_pose(self):
    #     self.full_map.fill_(0.)
    #     self.full_pose.fill_(0.)
    #     self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

    #     locs = self.full_pose.cpu().numpy()
    #     self.planner_pose_inputs[:, :3] = locs
    #     for e in range(self.num_scenes):
    #         r, c = locs[e, 1], locs[e, 0]
    #         loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
    #                         int(c * 100.0 / self.args.map_resolution)]

    #         self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

    #         self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
    #                                           (self.local_w, self.local_h),
    #                                           (self.full_w, self.full_h))

    #         self.planner_pose_inputs[e, 3:] = self.lmb[e]
    #         self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
    #                       self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

    #     for e in range(self.num_scenes):
    #         self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
    #         self.local_pose[e] = self.full_pose[e] - \
    #                         torch.from_numpy(self.origins[e]).to(self.map_gpu).float()

    def init_map_and_pose_for_env(self, e):
        self.full_map[e].fill_(0.)
        self.full_pose[e].fill_(0.)
        self.full_pose[e, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[e].cpu().numpy()
        self.planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                          (self.local_w, self.local_h),
                                          (self.full_w, self.full_h))

        self.planner_pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                      self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
        self.local_pose[e] = self.full_pose[e] - \
                        torch.from_numpy(self.origins[e]).to(self.map_gpu).float()

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if self.args.global_downscaling > 1:
                gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
                gx2, gy2 = gx1 + local_w, gy1 + local_h
                if gx1 < 0:
                    gx1, gx2 = 0, local_w
                if gx2 > full_w:
                    gx1, gx2 = full_w - local_w, full_w

                if gy1 < 0:
                    gy1, gy2 = 0, local_h
                if gy2 > full_h:
                    gy1, gy2 = full_h - local_h, full_h
            else:
                gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

            return [gx1, gx2, gy1, gy2]


class FILMNavInfoRetriever(InfoRetrieverBase):
    def __init__(self, nav: FILMNavigator) -> None:
        super().__init__(nav)
        self.last_subtask = None
        self.last_rgb = None
    
    def reset(self) -> None:
        self.last_subtask = None
        self.last_rgb = None

    def save_info(self, subtask: Subtask, rgb: np.ndarray) -> None:
        self.last_subtask = subtask
        self.last_rgb = rgb

    def update_predictor_values(self, subtask_success: bool) -> None:
        if self.last_subtask is None:
            return
        self.predictor.last_not_nav_subtask = self.last_subtask
        self.predictor.last_not_nav_subtask_success = subtask_success

        # self.picked_up_cat is not actually used in FILM, so it is not updated
        if self.last_subtask.action == 'PickupObject' and subtask_success:
            self.predictor.picked_up = True
            self.predictor.picked_up_mask = self.predictor.seg_model.get_interaction_mask(
                self.predictor.img_transform_seg(self.last_rgb),
                self.last_subtask.obj,
                use_area_as_score=True,
                check_zero_mask=False
            )
        elif self.last_subtask.action == 'PutObject' and subtask_success:
            self.predictor.picked_up = False
            self.predictor.picked_up_mask = None

        # The last condition is not present in FILM but was added due to possible retries:
        # if 'PutObject' is not successful, self.last_subtask.obj becomes a receptecle
        if (
            not self.predictor.args.no_pickup
            and self.predictor.args.no_pickup_update
            and self.predictor.picked_up
            and self.predictor.do_not_update_cat_s[0] is None
        ):
            self.predictor.do_not_update_cat_s[0] = \
                self.predictor.total_cat2idx[self.last_subtask.obj]
        elif not self.predictor.picked_up:
            self.predictor.do_not_update_cat_s[0] = None
