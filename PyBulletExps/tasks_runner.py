import os
import argparse

import numpy as np

from custom_logger import FileLogger
from motion_primitives import Motion_primitives
from table_environment import PickPlaceEnv
from tasks import (
    task1, 
    task2, 
    task3,
    task4,
    task5,
    task6,
    task7,
    task8,
    task9,
    task10
)
from utils import (
    display_images,
    is_task_successful,
    save_json,
    load_json,
    run_experiment,
    Replanner
)


def run_tasks(
    experiment_name: str, 
    tasks_from: int,
    tasks_to: int,
    model_name: str, 
    images_to_use: int, 
    images_step: int, 
    drop_prob: tuple, 
    runs_per_task: int, 
    save_video: bool,
    predict_steps: str,
    replanning_model: str
) -> None:
    # -------------------------------------------------------------------------------------------------
    # - Experiment
    # -------------------------------------------------------------------------------------------------
    # Create directory for logs if it does not exist
    log_dir = f"/home/mpatratskiy/work/pybullet/Pybullet/logs/{experiment_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = FileLogger(log_filename="log.txt", log_dir=log_dir, log_to_stdout=False)
    logger.set_log_level("INFO")
    
    logger.info(f"="*50)
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"="*50)
    base_path = f"/home/mpatratskiy/work/pybullet/Pybullet/results_logs/"
    experiment_path = os.path.join(base_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    replanner = Replanner(experiment_name, images_to_use=images_to_use, images_step=images_step, model_name=model_name, predict_steps=predict_steps, replanning_model=replanning_model)
    
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]
    tasks = tasks[tasks_from-1:tasks_to]
    
    for task in tasks:
        for exp_run in range(runs_per_task):
            random_seed = task["base_random_seed"] + exp_run
            np.random.seed(random_seed)
            if os.path.exists(os.path.join(experiment_path, task["name"] + f"_{exp_run}.json")):
                logger.info(f"Skipping task: {task['name']} + _{exp_run}, log already exists")
                continue
            logger.info(f"-"*50)
            logger.info(f"Running task: {task['name']}, goal: {task['goal']}")
            logger.info(f"-"*50)
            
            # Getting information about the task
            goal = task["goal"]
            obj_list = task["obj_list"]
            plan = task["plan"]
            validation_rule = task["validation_rule"]
        
            env = PickPlaceEnv(render=True)
            _ = env.reset(obj_list)
            motion = Motion_primitives(env, obj_list, drop_type='place', drop_prob=drop_prob)
            env_actions = {
                'locate': motion.locate,
                'pick': motion.pick,
                'place': motion.place,
                'done': lambda args=None: is_task_successful(env, validation_rule)
            }  

            available_actions = list(set([
                f"{action}('{obj}')" if action != 'done' else "done()"
                for action in env_actions
                for obj in obj_list
            ]))

            result_states = run_experiment(
                env=env, 
                replanner=replanner,
                obj_list=obj_list, 
                plan=plan, 
                validation_rule=validation_rule, 
                available_actions=available_actions, 
                env_actions=env_actions, 
                goal=goal, 
                logger=logger, 
                do_replan=do_replan, 
                max_replan_count=5
            )
            save_json(result_states, os.path.join(experiment_path, task["name"] + f"_{exp_run}.json"))
            if save_video:
                display_images(
                    images=env.cache_video[::13], 
                    save_path=os.path.join(experiment_path, task["name"] + f"_{exp_run}.gif"), 
                    do_display=False
                )



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiment with given configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Read the configuration file
    base_config_path = "/home/mpatratskiy/work/pybullet/Pybullet/exps_configs"
    config = load_json(os.path.join(base_config_path, args.config))

    # Fill args below with values from the configuration file
    experiment_name = config.get("experiment_name", "UNNAMED_EXPERIMENT")
    tasks_from = config.get("tasks_from", 0)
    tasks_to = config.get("tasks_to", 10)
    model_name = config.get("model_name", "gpt-4o")
    images_to_use = config.get("images_to_use", 1)
    images_step = config.get("images_step", 1)
    drop_prob = tuple(config.get("drop_prob", [0, 0.5]))
    do_replan = config.get("do_replan", True)
    runs_per_task = config.get("runs_per_task", 1)
    save_video = config.get("save_video", True)
    predict_steps = config.get("predict_steps", "lerp")
    replanning_model = config.get("replanning_model", "lera_api")
    run_tasks(experiment_name, tasks_from, tasks_to, model_name, images_to_use, images_step, drop_prob, runs_per_task, save_video, predict_steps, replanning_model)