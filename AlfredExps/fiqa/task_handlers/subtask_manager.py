from abc import ABCMeta, abstractmethod
import argparse
from typing import List, Tuple, Optional, Dict

import json
import re
import numpy as np
import requests
import torchvision.transforms as T

from fiqa.language_processing.subtask import Subtask
from fiqa.task_handlers.interactor import InteractorBase
from fiqa.perceivers.basics_and_dummies import SegModelBase
from fiqa.task_handlers.serialization import obj2str


class SubtaskManagerBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.subtask_queue = []
        self.subtask_ptr = 0

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def replan_subtasks(self) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def get_cur_subtask(self) -> Subtask:
        raise NotImplementedError()

    @abstractmethod
    def get_next_subtask(self) -> Tuple[Subtask, bool]:
        raise NotImplementedError()

    @abstractmethod
    def get_last_nav_subtask(self) -> Subtask:
        raise NotImplementedError()


class StaticQueueSubtaskManager(SubtaskManagerBase):
    """This subtask manager doesn't support changing the subtask queue."""

    def __init__(self) -> None:
        super().__init__()
        self.subtask_to_return_to = None

    def reset(self, subtask_queue: List[Subtask]):
        self.subtask_queue = subtask_queue
        self.subtask_ptr = 0
        self.subtask_to_return_to = None

    def replan_subtasks(self, rgb: np.ndarray, err: str = '') -> Dict:
        task_type = ''  # It is needed to determine only 'pick_two_...' tasks

        all_pickups_and_puts_with_goto = [
            subtask for subtask in self.subtask_queue
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
            task_type = 'pick_two_obj_and_place'

        update_info = {
            'subtask_queue': self.subtask_queue,  # No replan
            'task_type': task_type
        }
        return update_info

    def get_cur_subtask(self) -> Subtask:
        return self.subtask_queue[self.subtask_ptr]

    def get_next_subtask(self) -> Tuple[Subtask, bool]:
        if self.subtask_to_return_to is not None:
            self.subtask_ptr = self.subtask_to_return_to
            self.subtask_to_return_to = None
            task_finished = False
        else:
            task_finished = self.subtask_ptr == len(self.subtask_queue) - 1
            if not task_finished:
                self.subtask_ptr += 1
        return self.get_cur_subtask(), task_finished

    def get_last_nav_subtask(self) -> Subtask:
        for ind in range(self.subtask_ptr, -1, -1):
            if self.subtask_queue[ind].action == 'GotoLocation':
                if self.subtask_ptr != ind:
                    self.subtask_to_return_to = self.subtask_ptr
                    self.subtask_ptr = ind
                return self.get_cur_subtask()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LLServer and Relanner
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
from PIL import Image
import time

def save_images(images, dirs):
    for idx, (image, dir) in enumerate(zip(images, dirs)):
        Image.fromarray(image).save(dir)
    return dirs


class UniserverHandler:
    def __init__(self, port=8000):
        self.uniserver_port = port
        ping_result = self._ping_server()
        if ping_result["status"] == "Server is working":
            print("Server is ready to go")
        else:
            print(f"Error: {ping_result}")

    def get_running_models(self):
        url = f"http://host.docker.internal:{self.uniserver_port}/running_models/"
        response = requests.get(url)
        return response.json()

    def start_model(self, model_name: str):
        url = f"http://host.docker.internal:{self.uniserver_port}/start_model/"
        params = {
            'model_name': model_name
        }
        response = requests.post(url, params=params)
        return response.json()

    def stop_model(self, model_id):
        url = f"http://host.docker.internal:{self.uniserver_port}/stop_model/"
        params = {
            'model_id': model_id
        }
        response = requests.post(url, params=params)
        return response.json()
    
    def stop_all_models(self):
        models = self.get_running_models()["running_models"]["models"]
        for model_id in models.keys():
            res = self.stop_model(model_id)
            print(res)
            
    def put_task(self, model_id, prompt, image_paths, **kwargs):
        url = f"http://host.docker.internal:{self.uniserver_port}/put_task/"

        data = {
            'model_id': model_id,
            'prompt': prompt,
            'image_paths': image_paths,
            'extra_params': kwargs
        }
        print(kwargs)
        print(data)
        response = requests.post(url, json=data)
        return response.json()
    
    def get_task_result(self, model_id, task_id):
        url = f"http://host.docker.internal:{self.uniserver_port}/get_task_result/"
        params = {
            'model_id': model_id,
            'task_id': task_id
        }
        response = requests.post(url, params=params)
        return response.json()
    
    def _ping_server(self):
        """
        Function to ping the server and check if it is working.
        """
        url = f"http://host.docker.internal:{self.uniserver_port}/"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return {"status": "Server is working"}
            else:
                return {"status": "Server is not working", "code": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"status": "Server is not working", "error": str(e)}


# model_name="gemini-flash-1.5-8b"
# model_name="gpt-4o-mini"
# model_name="gpt-4o"
# model_name="gemini-pro-1.5"
# model = "llama-3.2-11b-vision-instruct"

class LLServerReplanner:
    def __init__(self, experiment_name, images_to_use=1, images_step=5, model_name="gpt-4o") -> None:
        self.handler = UniserverHandler(port=8000)
        start_response = self.handler.start_model("lera_baseline")
        self.model_id = start_response["model_id"]
        self.images_to_use = images_to_use
        self.images_step = images_step
        self.predict_steps = "ler"
        self.model_name = model_name + "###" + "alfred_2"
        self.save_base_dir = f"/home/mpatratskiy/data/{experiment_name}"
        self.load_base_dir = f"/llserver/alfred/data/{experiment_name}"
        os.makedirs(self.save_base_dir, exist_ok=True)
        print(self.save_base_dir)
        print(os.listdir(self.save_base_dir))
        self.goal_call_count = dict()
        
    def __del__(self):
        print(self.handler.stop_model(self.model_id))
        
    def replan(self, goal, success_actions, current_plan, all_images):
        self.goal_call_count[goal] = self.goal_call_count.get(goal, 0) + 1
        
        if self.goal_call_count[goal] > 3:
            # this will raise an error
            # but it is ok, because we want to stop replanning
            return None, None
        
        prompt = f"""{goal}###{success_actions}###{current_plan}###"""
        
        images = all_images[-(self.images_to_use-1)*self.images_step-1::self.images_step]
        dirs_to_save = [f"{self.save_base_dir}/{idx}.png" for idx in range(len(images))]
        dirs_to_load = [f"{self.load_base_dir}/{idx}.png" for idx in range(len(images))]
        save_images(images, dirs_to_save)
        
        put_response = self.handler.put_task(model_id=self.model_id, prompt=prompt, image_paths=dirs_to_load, model=self.model_name, predict_steps=self.predict_steps)
        task_id = put_response["task_id"]["task_id"]
        
        status = ""
        wait_time = 13
        while status != "completed":
            time.sleep(wait_time)
            result_response = self.handler.get_task_result(model_id=self.model_id, task_id=task_id)
            status = result_response.get("status")
            print(f"Status: {status}")
        
        if status == "not found":
            raise Exception("Task failed: Task not found")
        
        info = result_response.get("result")
        predicted_actions = info.get("[replan_response]")
        actions_list = predicted_actions.strip().strip("\n").strip('[]').split(', ')
        
        # Add field call_number into info
        info['call_number'] = self.goal_call_count[goal]
        # Save info into save_base_dir
        info_path = os.path.join(self.save_base_dir, f"info_{goal}_{self.goal_call_count[goal]}.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        return info, actions_list

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class NonStaticQueueSubtaskManager(StaticQueueSubtaskManager):
    """This subtask manager supports updating the subtask queue. Used with LLMs."""

    def __init__(self, seg_tuple: Tuple[SegModelBase, T.Compose], experiment_name: str = "UNNAMED_SUBTASK_MANAGER_EXPERIMENT") -> None:
        super().__init__()
        self.prev_action_plan = None
        self.seg_model, self.img_transform_seg = seg_tuple
        self.planner_url = "http://host.docker.internal:8089/"
        self.replanner = LLServerReplanner(experiment_name=experiment_name)
        
        self.map_model_alfred_actions = {
            "move": "GotoLocation",
            "slice": "SliceObject",
            "pick_up": "PickupObject",
            "put": "PutObject",
            "open": "OpenObject",
            "close": "CloseObject",
            "turn_on": "ToggleObjectOn",
            "turn_off": "ToggleObjectOff",
            "GotoLocation": "move",
            "SliceObject": "slice",
            "PickupObject": "pick_up",
            "PutObject": "put",
            "OpenObject": "open",
            "CloseObject": "close",
            "ToggleObjectOn": "turn_on",
            "ToggleObjectOff": "turn_off",
            "action": "action",
        }

    def _parse_action(self, step_text: str) -> Optional[Subtask]:
        """Parse action with arguments to step.
        text: put_on('pepper', 'white box')
        action: put_on
        arguments: ['pepper', 'white box']
        """
        step_decomposition_pattern = re.compile(r"\s*([A-Za-z_][A-Za-z_\s-]+)")
        args = step_decomposition_pattern.findall(step_text)

        print("----------")
        print(step_text)
        print(args)

        # To Add:
        # all was added

        if len(args) == 2:
            action = self.map_model_alfred_actions[args[0]]
            command_args = args[1].split()
            if command_args[0] == "slice" and command_args[1] == "of":
                command_args = command_args[2:]
                command_args.append("slice")
            elif "slices" in command_args:
                command_args = [command_args[0], "slice"]

            if "applesliced" in command_args:
                command_args = ["apple", "sliced"]

            if "stoveburner" in command_args:
                command_args = ["stove", "burner"]

            if "diningtable" in command_args:
                command_args = ["dining", "table"]

            if "breadsliced" in command_args:
                command_args = ["bread", "sliced"]

            if "tomatosliced" in command_args:
                command_args = ["tomato", "sliced"]

            if "microwave" in command_args or "oven" in command_args:
                command_args = ["microwave"]

            if "sink" in command_args or "dishwasher" in command_args:
                command_args = ["sink", "basin"]

            if "trash" in command_args or "bin" in command_args:
                command_args = ["trash", "bin"]

            if "garbage" in command_args or "garbagecan" in command_args:
                command_args = ["garbage", "can"]

            if "counter" in command_args or "countertop" in command_args:
                command_args = ["counter", "top"]

            if "freezer" in command_args:
                command_args = ["fridge"]

            if "kitchen" in command_args:
                command_args = ["table"]

            if "soap" in command_args:
                command_args = ["soap", "bar"]

            if "wine" in command_args:
                command_args = ["wine", "bottle"]

            if "tissue" in command_args:
                command_args = ["tissue", "box"]

            if "table" in command_args:
                command_args = ["side", "table"]

            if "remote" in command_args:
                command_args = ["remote", "control"]

            if "sponge" in command_args:
                command_args = ["dish", "sponge"]

            if "knife" in command_args:
                command_args = ["knife"]

            if "lamp" in command_args:
                command_args = ["desk", "lamp"]

            if action == "action" and ("".join(obj) not in ["move_behind", "right_side_step", "left_side_step", "move_until_visible"]):
                obj = "left_side_step"

            obj = "".join([
                "Sliced" if word == "slice" else word.capitalize()
                for word in command_args
            ])

            if "Cd" == obj:
                obj = "CD"

            step = Subtask((obj, action))
            print(step)
            return step
        elif len(args) == 3:
            action = self.map_model_alfred_actions[args[0]]
            obj = "".join([
                word.capitalize()
                for word in args[1].split()
            ])
            step = Subtask((obj, args[2], action))
            return step
        return None

    def get_subtask_queue(self, goal: str) -> List[Subtask]:
        url = self.planner_url + "add_llp_task"
        data = {
            "goal": " ".join([word.strip() for word in goal if word != "<<goal>>"]).replace(".", ""),
            "task_type": 1
        }

        print(f"Subtask queue data:     {data}"[:100])
        response = requests.post(url, json=data).json()
        print(f"Subtask queue response: {response}")

        steps = response["steps"].split("#")
        tasks = [self._parse_action(step) for step in steps]

        return tasks

    def get_help_for_nav(self, rgb: np.ndarray, subtask: Subtask) -> str:
        url = self.planner_url + "add_llp_task"
        image = obj2str(rgb)
        data = {
            # "goal": f"{subtask.action} {subtask.obj}, if you see <Img><ImageHere></Img>.",
            "goal": f"move to {subtask.obj}",
            "images": {"0": image},
            "task_type": 2
        }

        print(f"Help for nav data:     {data}"[:100])
        response = requests.post(url, json=data).json()
        print(f"Help for nav response: {response}")

        pose_correction = response["steps"][0]
        return pose_correction

    def replan_subtasks(self, rgb: np.ndarray, goal: List[str]) -> Dict:
        already_executed = None
        if self.subtask_to_return_to is not None:
            already_executed = \
                self.subtask_queue[self.subtask_ptr:self.subtask_to_return_to]

        print(f"{self.subtask_ptr=}")
        print(f"{self.subtask_to_return_to=}")
        print(f"{already_executed is None}")

        url = self.planner_url + "add_llp_task"
        image = obj2str(rgb)
        goal = " ".join([word.strip()
                        for word in goal if word != "<<goal>>"]).replace(".", "")
        action_plan = ", ".join([
            f'{self.map_model_alfred_actions[subtask.action]}("{subtask.obj.lower()}")'
            for subtask in self.subtask_queue[self.subtask_ptr + 1:]
        ])

        if self.prev_action_plan is None:
            self.prev_action_plan = action_plan
        else:
            if self.prev_action_plan == action_plan:
                raise ValueError("curr action plan and previous are the same")

        self.prev_action_plan = action_plan

        # replan_request = goal + "###" + action_plan

        # data = {
        #     "goal": replan_request,
        #     "images": {"0": image},
        #     "task_type": 3
        # }

        # print(f"Replan data:     {data}"[:100])
        # response = requests.post(url, json=data).json()
        # print(f"Replan response: {response}")

        goal = goal
        success_actions = ""
        if already_executed is not None:
            success_actions = ", ".join([
                f'{self.map_model_alfred_actions[subtask.action]}("{subtask.obj.lower()}")'
                for subtask in already_executed
            ])
        current_plan = action_plan
        all_images = [rgb]
        info, actions_list = self.replanner.replan(goal, success_actions, current_plan, all_images)
        steps = actions_list
        
        # TODO: Predict task_type
        # currently affect only on pick_two_obj_and_place task type
        task_type = 'NONE'
        # TODO: errors handling and even move to gen_method?
        # steps = response["steps"].split("#")

        if steps == ['']:
            self.subtask_queue = self.subtask_queue[:self.subtask_ptr+1]
            self.subtask_to_return_to = None
            info_for_update = {
                'subtask_queue': self.subtask_queue, 'task_type': task_type}
            return info_for_update
        else:
            tasks = [self._parse_action(step) for step in steps]

        print("Subtask queue:")
        print(", ".join([
            f'{self.map_model_alfred_actions[subtask.action]}("{subtask.obj.lower()}")'
            for subtask in self.subtask_queue
        ]))
        print("Curr plan:")
        print(action_plan)

        # skipping one already executed step
        # and one problematic step
        # self.subtask_ptr += 2
        # self.subtask_queue[self.subtask_ptr] = tasks[0]
        self.subtask_ptr += 2
        self.subtask_queue[self.subtask_ptr] = tasks[0]
        # skipping one already executed step
        # self.subtask_ptr += 1
        # self.subtask_queue[self.subtask_ptr:] = tasks

        print("New plan:")
        if steps != ['']:
            new_plan = ", ".join([
                f'{self.map_model_alfred_actions[subtask.action]}("{subtask.obj.lower()}")'
                for subtask in self.subtask_queue[self.subtask_ptr + 1:]
            ])
        else:
            new_plan = ""
        print(new_plan)

        if new_plan == action_plan:
            raise ValueError("same plan was predicted")

        shift = 0
        if already_executed is not None:
            while (
                self.subtask_ptr + shift <= self.subtask_to_return_to
                and shift < len(already_executed)
                and self.subtask_queue[self.subtask_ptr + shift] == already_executed[shift]
            ):
                shift += 1
        self.subtask_ptr += shift

        print("NEXT STEP: ")
        print(", ".join([
            f'{self.map_model_alfred_actions[subtask.action]}("{subtask.obj.lower()}")'
            for subtask in self.subtask_queue[self.subtask_ptr:]
        ]))

        self.subtask_to_return_to = None
        info_for_update = {
            'subtask_queue': self.subtask_queue, 'task_type': task_type}

        print(f"{self.subtask_ptr=}")
        print(f"{self.subtask_to_return_to=}")
        print(f"{info_for_update=}")

        return info_for_update


def build(
    args: argparse.Namespace, interactor: Optional[InteractorBase] = None
) -> SubtaskManagerBase:
    if args.planner == 'no_replan':
        return StaticQueueSubtaskManager()
    elif args.planner == 'with_replan':
        assert interactor is not None, 'An interactor must be provided!'
        # Currently we have only segmentation based interactors and they have
        # `seg_model` and `img_transform_seg` fields
        seg_tuple = (interactor.seg_model, interactor.img_transform_seg)
        experiment_name = args.run_name
        return NonStaticQueueSubtaskManager(seg_tuple, experiment_name=experiment_name)
    else:
        assert False, f'Unknown type {args.planner} of subtask manager'
