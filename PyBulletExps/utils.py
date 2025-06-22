import json
import os
import re
import time

import numpy as np
import requests
from IPython.display import display
from PIL import Image

from llserver.utils.handler import UniserverHandler
from table_environment import PickPlaceEnv

# # Адрес вашего FastAPI сервера
# BASE_URL = "http://127.0.0.1:8000"

# def put_task(image_paths, prompt):
#     """
#     Функция для добавления задачи с несколькими изображениями в очередь.
#     """
#     url = f"{BASE_URL}/put_task/"
    
#     # Превращаем список путей в строку с разделителем, чтобы передать через форму
#     data = {
#         'image_paths': image_paths,  # Передаём как строку с разделителем
#         'prompt': prompt
#     }
    
#     # Выполняем POST запрос
#     response = requests.post(url, data=data)
    
#     return response.json()

# def get_task_result(task_id):
#     """
#     Функция для получения результата задачи по её ID.
#     """
#     url = f"{BASE_URL}/get_task_result/"
#     json_data = {"task_id": task_id}
#     response = requests.post(url, json=json_data)
#     return response.json()


def save_image(image, name):
    base_path = "/home/mpatratskiy/work/pybullet/Pybullet/saved_images"
    save_path = f"{base_path}/{name}.png"
    Image.fromarray(image).save(save_path)
    return save_path

def save_images(images, dirs):
    for idx, (image, dir) in enumerate(zip(images, dirs)):
        Image.fromarray(image).save(dir)
    return dirs


def display_image(image):
    display(Image.fromarray(image))

    
def display_images(images, save_path=None, do_display=True):
    """Create a GIF from a list of RGB images. Save to the given path if provided, else return HTML to view."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure()
    ims = []

    for image in images:
        im = plt.imshow(image, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

    if save_path:
        ani.save(save_path, writer='imagemagick')
        plt.close(fig)  # Prevents the final frame from being displayed as a static image

    if do_display:
        plt.close(fig)  # Prevents the final frame from being displayed as a static image
        return HTML(ani.to_jshtml())


# Define a function to extract action name and arguments
def extract_action_details(action):
    actions_names_map = {
        "pick": "pick",
        "place_on_top_of": "place",
        "place": "place",
        "locate": "locate",
        "done": "done",
    }
    
    action = action.strip("'")
    action = action.strip('"')
    match = re.match(r"(\w+)\('?(.*?)'?\)", action)
    if match:
        action_name = match.group(1)
        action_args = match.group(2)
        return actions_names_map[action_name], action_args
    return None, None


def is_task_successful(env: PickPlaceEnv, validation_rule: list[tuple[str, str]]):
    observation = env.get_observation()
    reward = env.get_reward()
    done = True
    info = {
        "success": True,
        "success_conditions": []
    }
    for (obj_a, obj_b) in validation_rule:
        if not env.on_top_of(obj_a, obj_b):
            info["success"] = False
        else:
            info["success_conditions"].append((obj_a, obj_b))
    return observation, reward, done, info


def validate_action_execution(env: PickPlaceEnv, action: str, state_info: dict, action_info: dict):
    action_name, object_name = extract_action_details(action)
    
    if action_name == 'done':
        return is_task_successful(env, state_info["validation_rule"])[3]["success"]
    
    if action_name == "place":
        if state_info["gripper_object"] is None:
            return False
        return env.on_top_of(state_info["gripper_object"], object_name)
    
    if action_name == 'locate':
        location = action_info["location"]
        return location is not None
    
    if action_name == "pick":
        # add pick check - if pick is successful, then we need to check if the object is in the gripper
        return True
    
    raise ValueError(f"Unknown action: {action}")


def save_json(data, filename):
    def default_converter(o):
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(filename, 'w') as f:
        json.dump(data, f, default=default_converter)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
    
def validate_plan(plan_actions, available_actions):
    for action in plan_actions:
        action = action.replace("place_on_top_of", "place")
        action = action.strip("'")
        action = action.strip('"')
        if action not in available_actions:
            print(f"Action {action} not in available actions")
            return False
    return True


class Replanner:
    def __init__(self, experiment_name, images_to_use=1, images_step=5, model_name="gemini-pro-1.5", predict_steps="ler", replanning_model="lera_api") -> None:
        self.handler = UniserverHandler(port=8000)
        start_response = self.handler.start_model(replanning_model)
        self.model_id = start_response["model_id"]
        self.images_to_use = images_to_use
        self.images_step = images_step
        self.model_name = model_name
        self.predict_steps = predict_steps
        self.save_base_dir = f"/home/mpatratskiy/work/meta_world/llserver/data/{experiment_name}"
        self.load_base_dir = f"/llserver/data/{experiment_name}"
        os.makedirs(self.save_base_dir, exist_ok=True)
        
    def __del__(self):
        print(self.handler.stop_model(self.model_id))
        
    def replan(self, goal, success_actions, current_plan, available_actions, all_images):
        # TODO: unify convertion from prompted command to actually executed
        prompt = f"""{goal}###{success_actions}###{current_plan}###{available_actions}""".replace("place", "place_on_top_of") 
        
        dirs_to_load = []
        if self.images_to_use > 0:
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
        return info, actions_list


def run_experiment(env, replanner, obj_list, plan, validation_rule, available_actions, env_actions, goal, logger, do_replan=False, max_replan_count=5):
    objects_location = {
        obj: env_actions["locate"](obj)[3]["location"]
        for obj in obj_list
    }

    # Get image
    images = []
    image = env.get_camera_image()
    images.append(image)
    logger(image)

    state_logs = []
    plan_actions = plan.split('\n')
    idx = 0
    
    inputs = {
        "goal": goal,
        "obj_list": obj_list,
        "plan": plan,
        "validation_rule": validation_rule,
        "available_actions": available_actions,
    }
    
    state_info = {
        "inputs": inputs,
        "action_idx": None,
        "action": None,
        "action_name": None,
        "object_name": None,
        "success": False,
        "done": False,
        "last_action_success": False,
        "validation_rule": validation_rule,
        "gripper_object": None,
        "plan_actions": plan_actions,
        "replan_count": 0,
        "was_replanning_successful": True,
    }

    while idx < len(plan_actions):
        # Get action
        action = plan_actions[idx]
        action_name, object_name = extract_action_details(action)
        logger(f"Action Name: '{action_name}', Action Arguments: '{object_name}' ")
        
        state_info["plan_actions"] = plan_actions
        state_info["action_idx"] = idx
        state_info["action"] = action
        state_info["action_name"] = action_name
        state_info["object_name"] = object_name
        
        # Execute action
        obs, reward, done, info = None, None, None, None
        if action_name == 'locate':
            action_args = object_name
        elif action_name == 'pick':
            action_args = objects_location[object_name]
        elif action_name == 'place':
            action_args = objects_location[object_name]
        elif action_name == 'done':
            action_args = None
            
        logger(f"State info before action: {state_info}")
        obs, reward, done, info = env_actions[action_name](action_args) # TODO: get image from observation
        state_info["last_action_success"] = validate_action_execution(env, action, state_info, info)
        
        if action_name == 'locate':
            objects_location[object_name] = info["location"]
            logger(f"Location of {object_name}: {info['location']}")
            
        elif action_name == 'pick':
            state_info["gripper_object"] = object_name

        elif action_name == 'place':
            state_info["gripper_object"] = None
            
        elif action_name == 'done':
            _, _, _, info = is_task_successful(env, state_info["validation_rule"])
            state_info["success"] = info["success"]
            state_info["success_conditions"] = info["success_conditions"]
            state_info["done"] = True

        logger(f"State info after action: {state_info}")
        image = env.get_camera_image()
        images.append(image)
        logger(image)

        if not state_info["last_action_success"]:
            if do_replan:
                if state_info["replan_count"] >= max_replan_count:
                    logger("Max replan count reached. Stopping...")
                    done = True
                else:
                    state_info["replan_count"] += 1
                    logger(f"Action '{action_name}' failed. Replanning...")
                    info, plan_actions = replanner.replan(goal, plan_actions[:idx], plan_actions[idx:], available_actions, env.cache_video)
                    logger(f"New plan: {plan_actions}")
                    is_plan_valid = validate_plan(plan_actions, available_actions)
                    if not is_plan_valid:
                        logger("Plan is not valid. Stopping...")
                        state_info["was_replanning_successful"] = False
                        state_info["replan_logs"] = info
                        done = True
                    else:
                        logger("Plan is valid. Continuing...")
                        state_info["was_replanning_successful"] = True
                        state_info["replan_logs"] = info
                        done = False
                
                idx = -1
            else:
                logger(f"Action '{action_name}' failed. Stopping...")
                done = True

        if done:
            _, _, _, info = is_task_successful(env, state_info["validation_rule"])
            state_info["success"] = info["success"]
            state_info["success_conditions"] = info["success_conditions"]
        
        state_logs.append(state_info.copy())
        
        if done:
            break
        
        # Increment index
        idx += 1
    logger(f"Task completed. Success: {state_info['success']}")

    return state_logs


def calc_metrics(experiment_name, logs_folder='results_logs'):
    def parse_log_filename(filename):
        # Use regex to extract the task name and number
        match = re.match(r"(.*)_(\d+)\.json", filename)
        if match:
            task_name = match.group(1)
            task_number = match.group(2)
            return task_name, task_number
        return None, None
    
    experiment_folder = os.path.join(logs_folder, experiment_name)
    logs = [f for f in os.listdir(experiment_folder) if f.endswith('.json')]
    tasks_names = set([parse_log_filename(log)[0] for log in logs])
    tasks2logs = {task_name: [] for task_name in tasks_names}
    for log in logs:
        task_name, _ = parse_log_filename(log)
        tasks2logs[task_name].append(log)
    for key in tasks2logs:
        tasks2logs[key].sort()

    metrics = {}
    for task_name, logs in tasks2logs.items():
        metrics[task_name] = {}
        
        total_replan_count = 0
        total_number_of_successes = []
        total_actions_made = []
        total_mean_place_actions_failed = []
        total_valid_replanning = []
        tasks_with_replanning = []
        total_place_actions = []
        total_goal_conditioned_success = []
        total_validation_rules_count = []
        success_rates = []
        replan_success_rates = []
        goal_conditioned_success_rates = []
        
        for log in logs:
            data = load_json(os.path.join(experiment_folder, log))
            total_replan_count += data[-1]["replan_count"]
            total_number_of_successes.append(data[-1]["success"])
            total_actions_made.append(len(data))
            total_actions_failed = sum(1 for entry in data if not entry["last_action_success"])
            total_place_actions.append(sum(1 for entry in data if entry["action_name"] == "place"))

            total_mean_place_actions_failed.append(total_actions_failed / total_place_actions[-1])
            total_valid_replanning.append(sum(
                1 
                for idx in range(1, len(data)) 
                if data[idx]["was_replanning_successful"] and 
                (data[idx]["replan_count"] - data[idx-1]["replan_count"]) > 0
            ))
            try:
                total_goal_conditioned_success.append(len(data[-1]["success_conditions"]))
            except:
                total_goal_conditioned_success.append(0)
                
                print(f"Error in {log}")
            total_validation_rules_count.append(len(data[-1]["validation_rule"]))
            tasks_with_replanning.append(1 if data[-1]["replan_count"] > 0 else 0)
            
            # Calculate individual success rates for std calculation
            success_rates.append(data[-1]["success"])
            if sum(tasks_with_replanning) > 0:
                replan_success_rates.append(data[-1]["success"])
            
        
        tasks_without_errors = len(logs) - sum(tasks_with_replanning)
        total_number_of_successes_and_replanning = sum(total_number_of_successes) - tasks_without_errors
        
        # Average by number of task runs
        metrics[task_name]["runs number"] = len(logs)
        metrics[task_name]["runs success"] = sum(total_number_of_successes)
        metrics[task_name]["success rate"] = sum(total_number_of_successes) / len(logs) * 100
        metrics[task_name]["goal conditioned success"] = sum(total_goal_conditioned_success) / sum(total_validation_rules_count) * 100
        metrics[task_name]["tasks with replanning"] = sum(tasks_with_replanning)
        metrics[task_name]["tasks with replanning success"] = total_number_of_successes_and_replanning if sum(tasks_with_replanning) > 0 else 0
        metrics[task_name]["replan success"] = total_number_of_successes_and_replanning / sum(tasks_with_replanning) * 100 if sum(tasks_with_replanning) > 0 else 0
        metrics[task_name]["replan number"] = total_replan_count
        metrics[task_name]["mean_replan_count"] = total_replan_count / len(logs)
        metrics[task_name]["mean_actions_made"] = sum(total_actions_made) / len(logs)
        metrics[task_name]["mean_place_actions_failed"] = sum(total_mean_place_actions_failed) / len(logs)
        metrics[task_name]["mean_valid_replanning"] = sum(total_valid_replanning) / len(logs)
        
        # Calculate standard deviations
        metrics[task_name]["success rate std"] = np.std(success_rates) if success_rates else 0
        metrics[task_name]["replan success std"] = np.std(replan_success_rates) if replan_success_rates else 0
        metrics[task_name]["goal conditioned success std"] = np.std(goal_conditioned_success_rates) if goal_conditioned_success_rates else 0
        

    metrics["total"] = {
        metric: sum(
            metrics[task_name][metric] 
            for task_name in tasks_names
        ) 
        if metric not in ["success rate", "replan success", "goal conditioned success"]
        else sum(
            metrics[task_name][metric] 
            for task_name in tasks_names
        ) / len(tasks_names)
        for metric in metrics[task_name]
    }
    metrics["total"]["success rate std"] = np.std([
        metrics[task_name]["success rate"]
        for task_name in tasks_names
    ])
    metrics["total"]["replan success std"] = np.std([
        metrics[task_name]["replan success"]
        for task_name in tasks_names
    ])
    metrics["total"]["goal conditioned success std"] = np.std([
        metrics[task_name]["goal conditioned success"]
        for task_name in tasks_names
    ])
    
    # Sort metrics by alphabetical order
    metrics = {k: metrics[k] for k in sorted(metrics)}
    return metrics


def compare_experiment_results(experiment_name1, experiment_name2, logs_folder='results_logs'):
    def parse_log_filename(filename):
        # Use regex to extract the task name and number
        match = re.match(r"(.*)_(\d+)\.json", filename)
        if match:
            task_name = match.group(1)
            task_number = match.group(2)
            return task_name, task_number
        return None, None

    def get_experiment_results_detailed(experiment_name):
        experiment_folder = os.path.join(logs_folder, experiment_name)
        logs = [f for f in os.listdir(experiment_folder) if f.endswith('.json')]
        task_results = {}
        for log in logs:
            task_name, task_number = parse_log_filename(log)
            data = load_json(os.path.join(experiment_folder, log))
            if task_name not in task_results:
                task_results[task_name] = {}
            task_results[task_name][task_number] = data[-1]["success"]
            # print(f"{experiment_name} | Task {task_name} {task_number} success: {data[-1]['success']}")
        return task_results

    results1 = get_experiment_results_detailed(experiment_name1)
    results2 = get_experiment_results_detailed(experiment_name2)

    all_tasks = set(results1.keys()).union(set(results2.keys()))
    comparison = []
    for task in all_tasks:
        task_runs = set(results1.get(task, {}).keys()).union(set(results2.get(task, {}).keys()))
        for run in task_runs:
            result1 = results1.get(task, {}).get(run, 0)  # Default to 0 if run not found
            result2 = results2.get(task, {}).get(run, 0)  # Default to 0 if run not found
            comparison.append((f"{task}_{run}", result1, result2))
    # Sort the comparison by task names
    comparison.sort(key=lambda x: x[0])
    return comparison