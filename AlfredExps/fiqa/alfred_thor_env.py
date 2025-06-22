import argparse
from typing import Any
from pathlib import Path
import pickle

from alfred_utils.env.thor_env import ThorEnv


class AlfredThorEnv:
    """Helper class for using ThorEnv inside the model's modules."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.thor_env = ThorEnv()  # Start THOR
        if args.subdataset_type == 'changing_states':
            self.cats_to_change_states = [
                'Fridge', 'Microwave', 'DeskLamp', 'FloorLamp'
            ]  # Sliceable objects and "Faucet" are not considered for now
            instr_type = f'{args.split}_instructions_processed_no_recept_alfred.p'
            instrs_path = Path(
                f'fiqa/language_processing/processed_instructions/{instr_type}'
            )
            assert instrs_path.is_file(), f'Instructions {instr_type} are needed!'
            with instrs_path.open(mode='rb') as f:
                self.no_recep_instrs = pickle.load(f)

    def reset(self, traj_data: dict):
        """Initializes the scene and agent from the task info."""
        # Setup the scene
        reward_type = 'dense'
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']
        r_idx = traj_data['repeat_idx']
        scene_name = 'FloorPlan%d' % scene_num
        self.thor_env.reset(scene_name)
        self.thor_env.restore_scene(
            object_poses, object_toggles, dirty_and_empty
        )

        # Initialize to the start position
        self.thor_env.step(dict(traj_data['scene']['init_action']))

        # Print the task description
        task_descr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        print(f"Task: {task_descr}")

        # Setup the task for a reward
        if 'tests' not in self.args.split:
            self.thor_env.set_task(
                traj_data, self.args, reward_type=reward_type
            )

        # Change the states of the target objects
        if self.args.change_states:
            print('Changing states!')
            cats_in_gt_plan = set(
                instr.obj
                for instr in self.no_recep_instrs[
                    traj_data['task_id'], traj_data['repeat_idx']
                ]
            )
            subtask_queue = set(
                (instr.action, instr.obj)
                for instr in self.no_recep_instrs[
                    traj_data['task_id'], traj_data['repeat_idx']
                ]
            )
            for obj in self.thor_env.last_event.metadata['objects']:
                if obj['objectType'] in cats_in_gt_plan:
                    if (
                        obj['openable']
                        and obj['objectType'] in self.cats_to_change_states
                    ):
                        self.thor_env.step(dict(
                            action='OpenObject', objectId=obj['objectId'],
                            forceVisible=True, forceAction=True
                        ))
                    elif (
                        obj['toggleable']
                        and obj['objectType'] in self.cats_to_change_states
                    ):
                        self.thor_env.step(dict(
                            action='ToggleObjectOn', objectId=obj['objectId'],
                            forceVisible=True, forceAction=True
                        ))
                    # if (
                    #     obj['sliceable']
                    #     and ('SliceObject', obj['objectType']) in subtask_queue
                    # ):
                    #     self.thor_env.step(dict(
                    #         action='SliceObject', objectId=obj['objectId'],
                    #         forceVisible=True, forceAction=True
                    #     ))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.thor_env, name)
