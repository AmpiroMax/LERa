from fiqa.language_processing import subtasks_helper
from fiqa.task_handlers.planner import Planner
from fiqa.task_handlers.subtask_checker import build as build_subtask_checker
from fiqa.task_handlers.subtask_executor import SubtaskExecutor
from utils.logger import logger
from arguments import parse_args
import torch
import numpy as np
import pickle
import json
import random
import os
from tqdm import trange
os.environ['PYTHONHASHSEED'] = str(1)  # for the reproducibility


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # Fix the seed for reproducibility.
    set_seed(args.seed)
    # Set torch to deterministic mode for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
    if 'film' in args.navigator:
        print(
            'WARNING! Deterministic behaviour of the FILM navigator is not '
            + 'guaranteed since it uses torch.Tensor.scatter_add_() inside the'
            + 'Semantic_Mapping class and the torch version is too old.'
        )

    # In that script, we only use pretrained modules
    torch.set_grad_enabled(False)

    # Create directories for logging
    if not os.path.exists(f'results/{args.split}'):
        os.makedirs(f'results/{args.split}', exist_ok=True)
    os.chdir(f'results/{args.split}')
    os.mkdir(f'{args.run_name}')  # Force to set a unique 'run_name'
    os.chdir(f'{args.run_name}')
    if args.save_imgs:
        for episode_idx in range(args.from_idx, args.to_idx + 1):
            os.makedirs(f'images/{episode_idx}', exist_ok=True)
    os.chdir('../../../')
    logger.set_log_dir(f'results/{args.split}/{args.run_name}')

    # Load scene names from the chosen data split
    scene_names = subtasks_helper.load_scene_names(
        args.split, args.path_to_scene_names, args.dataset
    )

    # Create the subtask executor, checker and planner from the article
    subtask_executor = SubtaskExecutor(args)
    alfred_env = None if 'oracle' not in args.checker else subtask_executor.env
    subtask_checker = build_subtask_checker(
        args, subtask_executor.interactor, alfred_env)
    planner = Planner(args, subtask_executor.interactor)

    # Main part
    actseqs = []
    analyze_recs = []
    used_eps = set()
    if args.subdataset_type == 'changing_states':
        with open(
            f'alfred_utils/data/filtered_{args.split}_eps_for_changing_states.json',
            'r'
        ) as f:
            task_type2eps = json.load(f)
            used_eps = set(
                vv
                for v in task_type2eps[args.split].values()
                for vv in v.keys()
            )
    elif args.subdataset_type == 'pose_corrections':
        pose_correction_types = [
            'left_side_step', 'right_side_step', 'move_behind', 'move_until_visible'
        ]
        for pose_correction_type in pose_correction_types:
            pose_corr_metas_folder = os.path.join(
                'fiqa/language_processing/dataset_for_pose_correction',
                pose_correction_type, 'metas'
            )
            for entry in sorted(
                map(lambda x: x.path, os.scandir(pose_corr_metas_folder))
            ):
                with open(entry, 'r') as f:
                    pose_corr = json.load(f)
                used_eps.add(str(tuple(pose_corr['task_id_and_r_idx'])))

    for episode_idx in trange(args.from_idx, args.to_idx + 1):
        for change_states in [True, False]:
            args.change_states = change_states
            try:
                print(
                    f'Starting episode #{episode_idx} with '
                    + f'change_states = {change_states}'
                )
                # Load scene data from ALFRED annotation files by the current
                # scene name and extract subtasks by task_id and repeat_index
                traj_data = subtasks_helper.load_traj(
                    scene_names[episode_idx], args.lp_data, args.dataset
                )
            
                task_id, r_idx = traj_data['task_id'], traj_data['repeat_idx']
                if len(used_eps) and str((task_id, r_idx)) not in used_eps:
                    print('Skipping due to absence in the dataset')
                    continue
                

            
                instructions_processed = subtasks_helper.load_processed_instructions(
                    args.split, args.instr_type, args.dataset,
                    args.path_to_instructions, args.use_gt_instrs
                )
                task_info = traj_data['ann']['goal']
                subtask_queue = instructions_processed[(task_id, r_idx)]
            

                # Reset the env with a new scene and reset the main modules
                subtask_executor.reset(traj_data, subtask_queue)
                subtask_checker.reset()
                planner.reset(subtask_queue)
                logger.reset(episode_idx)

                # We should set the seed every time to try to ensure that the results of
                # a particular episode don't depend on its actual position in the range
                shift = episode_idx + change_states * 10000 if not args.debug else 0
                set_seed(args.seed + shift)

                # The first subtask should always be navigation
                task_finished = False
                subtask = planner.get_cur_subtask()
                retry_subtask = False
                steps_taken = 0
                help_cnt = 0
                while not task_finished:
                    while subtask.action == 'GotoLocation' and not task_finished:
                        pose_correction_type = None
                        if (
                            args.navigator == 'film'
                            and not args.film_use_stop_analysis
                            and args.planner == 'with_replan'
                            and retry_subtask
                            and help_cnt < 1
                        ):
                            pose_correction_type = planner.get_help_for_nav(
                                rgb, subtask)
                            help_cnt += 1
                        steps_taken_before = steps_taken
                        rgb, steps_taken = subtask_executor.execute_nav_subtask(
                            subtask, retry_subtask, pose_correction_type
                        )
                        # subtask, info_for_update = planner.replan(rgb, task_info)

                        # TODO: enable FILM to refuse to execute
                        if retry_subtask and steps_taken == steps_taken_before:
                            # Nav refused to execute
                            # TODO: better errors
                            subtask, info_for_update = planner.replan(
                                rgb, task_info)

                            task_finished, retry_subtask = False, False
                            subtask_executor.update_states(info_for_update)
                            help_cnt = 0
                        else:
                            verdict = subtask_checker.check(
                                rgb, subtask, steps_taken)
                            subtask, task_finished, retry_subtask = \
                                planner.plan_for_nav(verdict)

                    while subtask.action != 'GotoLocation' and not task_finished:
                        steps_taken_before = steps_taken
                        rgb, steps_taken = subtask_executor.execute_interaction_subtask(
                            subtask, retry_subtask
                        )
                        if steps_taken == steps_taken_before:
                            # Interactor refused to execute
                            subtask, task_finished, retry_subtask = \
                                planner.plan_for_nav(checker_verdict=False)
                        else:
                            verdict = subtask_checker.check(
                                rgb, subtask, steps_taken)
                            subtask, task_finished, retry_subtask = \
                                planner.plan_for_interaction(verdict)
                            if verdict:
                                help_cnt = 0
                                subtask_executor.reset_interactor_refuse_cnt()
            except AssertionError as e:
                print(
                    f'Episode {episode_idx} failed with assertion error: {e}')
                # Consider the episode as failed in subtask_executor.evaluate()
                subtask_executor.fails = args.max_fails
            except Exception as e:
                print(e)
                # continue
                # Consider the episode as failed in subtask_executor.evaluate()
                subtask_executor.fails = args.max_fails

            # continue
            
            actseqs.append({task_id: subtask_executor.actseq})
            # Save action sequences to upload on the evaluation server, if test
            if 'tests' in args.split:
                if args.split == 'tests_unseen':
                    results = {'tests_seen': [], 'tests_unseen': actseqs}
                else:
                    results = {'tests_seen': actseqs, 'tests_unseen': []}
                log_dir = f'results/{args.split}/{args.run_name}/'
                with open(log_dir + 'for_leaderboard.json', 'w') as r:
                    json.dump(results, r, indent=4, sort_keys=True)
            # Save successes and metrics for evaluation
            if 'tests' not in args.split:
                log_entry, success = subtask_executor.evaluate(
                    traj_data, r_idx, subtask_executor.steps_taken
                )
                print('Success is', success)
                logger.log_msg(f'Success is {success}!')
                print('Log entry is', str(log_entry))
                analyze_dict = {'success': success, 'log_entry': log_entry}
                analyze_recs.append(analyze_dict)
                pickle.dump(
                    analyze_recs,
                    open(
                        f'results/{args.split}/{args.run_name}/stats_recs.p', 'wb')
                )

    # Stop THOR
    subtask_executor.stop_env()
    print("All finished!")


if __name__ == '__main__':
    arguments = parse_args()
    print('Arguments:', arguments)
    main(arguments)
