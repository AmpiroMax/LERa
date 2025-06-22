import argparse
from glob import glob
from itertools import product
import json
import os
from PIL import Image
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict

from fiqa.language_processing import subtasks_helper
from fiqa.language_processing.subtask import Subtask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyzer of runs stats')
    parser.add_argument(
        '--split', type=str,
        choices=['train', 'valid_unseen', 'valid_seen'],
        required=True,
        help='ALFRED data split'
    )
    parser.add_argument(
        '--run_name_prefix', type=str,
        required=True,
        help='Prefix of the name of the directory inside results/[data_split]/'
        + ' where the logs are saved.'
    )
    parser.add_argument(
        '--run_name_postfix', type=str,
        required=True,
        help='Postfix of the name of the directory inside '
        + 'results/[data_split]/ where the logs are saved.'
    )
    parser.add_argument(
        '--path_to_scene_names', type=str, default='alfred_utils/data/splits',
        help='Path to the oct21.json folder'
    )
    parser.add_argument(
        '--lp_data', type=str, default='alfred_utils/data',
        help='Path to json_2.1.0 folder'
    )
    parser.add_argument(
        '--path_to_instructions', type=str,
        default='fiqa/language_processing/processed_instructions',
        help='Path to preprocessed instructions folder'
    )

    return parser.parse_args()


def save_rgb_and_metainfo(
    rgb_path: str, goal: str, task_type: str, plan: List[Subtask], cur_subtask: Tuple,
    pose_correction_type: str, task_id_and_r_idx: tuple, err: str = ''
):
    # RGB
    rgb = Image.open(rgb_path)
    rgb_idx = len(
        os.listdir(os.path.join(dataset_folder, pose_correction_type, image_folder))
    )
    rgb.save(os.path.join(
        dataset_folder, pose_correction_type, image_folder,
        f"{rgb_idx}_{rgb_path.split('/')[4]}.png"
    ))

    # Metainfo
    goal = str.join(' ', map(str.strip, goal[:-1]))
    metainfo = {
        'rgb_idx': f"{rgb_idx}_{rgb_path.split('/')[4]}.png",
        'goal': goal,
        'task_type': task_type,
        'plan': list(map(lambda x: (x.action, x.obj), plan)),
        'cur_subtask': cur_subtask,
        'err': err,
        'pose_correction_type': pose_correction_type,
        'task_id_and_r_idx': task_id_and_r_idx
    }
    meta_idx = len(
        os.listdir(os.path.join(dataset_folder, pose_correction_type, meta_folder))
    )
    meta_path = os.path.join(
        dataset_folder, pose_correction_type, meta_folder, f'{meta_idx}.json'
    )
    with open(meta_path, 'w') as f:
        json.dump(metainfo, f)


if __name__ == '__main__':
    args = parse_args()

    regexp = f'_[0-9]*_[0-9]*_' if args.run_name_postfix else f'_[0-9]*_[0-9]*'
    results_glob = glob(
        'results/' + args.split + '/' + args.run_name_prefix
        + regexp + args.run_name_postfix
    )
    if not results_glob:
        assert False, 'Logs with such names do not exist!'

    dataset_folder = 'fiqa/language_processing/dataset_for_pose_correction'
    image_folder = 'images'
    meta_folder = 'metas'
    pose_correction_types = [
        'left_side_step', 'right_side_step', 'move_behind', 'move_until_visible'
    ]
    for folder in product(pose_correction_types, [image_folder, meta_folder]):
        if not os.path.exists(os.path.join(dataset_folder, *folder)):
            os.makedirs(os.path.join(dataset_folder, *folder), exist_ok=True)

    # Load scene names from the chosen data split
    scene_names = subtasks_helper.load_scene_names(
        args.split, args.path_to_scene_names, 'alfred'
    )
    # Load processed instructions from the file
    instructions_processed = subtasks_helper.load_processed_instructions(
        args.split, 'film', 'alfred', args.path_to_instructions, False
    )

    success_eps = defaultdict(list)
    for target_dir in sorted(results_glob):
        if '0_819' in target_dir:
            continue
        for entry in tqdm(os.scandir(target_dir), desc=f'{target_dir}'):
            if not (entry.is_file() and entry.name[-4:] == '.txt'):
                continue
            with open(entry.path, 'r') as f:
                logs = f.readlines()[2:]

            if logs[-1].split()[-1][:-1] == 'False':
                continue

            traj_data = subtasks_helper.load_traj(
                scene_names[int(entry.name[:-4])], args.lp_data, 'alfred'
            )
            task_id, r_idx = traj_data['task_id'], traj_data['repeat_idx']
            filtered_subtasks = list()
            for j in range(0, len(logs) - 1):
                if (
                    'Warning' in logs[j] or 'Error' in logs[j] or logs[j][0] == ' '
                ):
                    continue
                splitted = logs[j].split()
                subtask_type = splitted[2][1:-1]
                subtask_target = splitted[3][:-1]
                action = splitted[5]
                steps_taken = int(splitted[6])
                # success = splitted[7]
                filtered_subtasks.append(
                    (subtask_type, subtask_target, action, steps_taken)
                )

                if (
                    len(filtered_subtasks) > 3
                    and ((
                        'StopNav' in filtered_subtasks[-1][2]
                        and 'RotateLeft' in filtered_subtasks[-2][2]
                        and 'MoveAhead' in filtered_subtasks[-3][2]
                        and 'RotateRight' in filtered_subtasks[-4][2]
                    ) or (
                        'StopNav' in filtered_subtasks[-1][2]
                        and 'RotateRight' in filtered_subtasks[-2][2]
                        and 'MoveAhead' in filtered_subtasks[-3][2]
                        and 'RotateLeft' in filtered_subtasks[-4][2]
                    ))
                ):
                    rgb_path = os.path.join(
                        target_dir, 'images', entry.name[:-4], f'{steps_taken - 3}.png'
                    )
                    if 'RotateRight' in filtered_subtasks[-4][2]:
                        if int(entry.name[:-4]) not in success_eps['right_side_step']:
                            success_eps['right_side_step'].append(int(entry.name[:-4]))
                    else:
                        if int(entry.name[:-4]) not in success_eps['left_side_step']:
                            success_eps['left_side_step'].append(int(entry.name[:-4]))
                    save_rgb_and_metainfo(
                        rgb_path, traj_data['ann']['goal'], traj_data['task_type'],
                        instructions_processed[(task_id, r_idx)],
                        cur_subtask=(subtask_type, subtask_target),
                        err='',  # TODO: what to write?
                        pose_correction_type=(
                            'right_side_step'
                            if 'RotateRight' in filtered_subtasks[-4][2]
                            else 'left_side_step'
                        ),
                        task_id_and_r_idx=(task_id, r_idx)
                    )
                if (
                    len(filtered_subtasks) > 5
                    and 'StopNav' in filtered_subtasks[-1][2]
                    and 'RotateLeft' in filtered_subtasks[-2][2]
                    and 'RotateLeft' in filtered_subtasks[-3][2]
                    and 'MoveAhead' in filtered_subtasks[-4][2]
                    and 'RotateLeft' in filtered_subtasks[-5][2]
                    and 'RotateLeft' in filtered_subtasks[-6][2]
                ):
                    rgb_path = os.path.join(
                        target_dir, 'images', entry.name[:-4], f'{steps_taken - 5}.png'
                    )
                    if int(entry.name[:-4]) not in success_eps['move_behind']:
                        success_eps['move_behind'].append(int(entry.name[:-4]))
                    save_rgb_and_metainfo(
                        rgb_path, traj_data['ann']['goal'], traj_data['task_type'],
                        instructions_processed[(task_id, r_idx)],
                        cur_subtask=(subtask_type, subtask_target),
                        err='',  # TODO: what to write?
                        pose_correction_type='move_behind',
                        task_id_and_r_idx=(task_id, r_idx)
                    )
                if (
                    len(filtered_subtasks) > 5
                    and 'LookUp' in filtered_subtasks[-1][2]
                    and 'RotateLeft' in filtered_subtasks[-2][2]
                    and 'RotateLeft' in filtered_subtasks[-3][2]
                    and 'RotateLeft' in filtered_subtasks[-4][2]
                    and 'RotateLeft' in filtered_subtasks[-5][2]
                    and 'RotateLeft' in filtered_subtasks[-6][2]
                ):
                    rgb_path = os.path.join(
                        target_dir, 'images', entry.name[:-4], f'{steps_taken - 6}.png'
                    )
                    if int(entry.name[:-4]) not in success_eps['move_until_visible']:
                        success_eps['move_until_visible'].append(int(entry.name[:-4]))
                    save_rgb_and_metainfo(
                        rgb_path, traj_data['ann']['goal'], traj_data['task_type'],
                        instructions_processed[(task_id, r_idx)],
                        cur_subtask=(subtask_type, subtask_target),
                        err='',  # TODO: what to write?
                        pose_correction_type='move_until_visible',
                        task_id_and_r_idx=(task_id, r_idx)
                    )

    print('Finished!')
    print(len(success_eps['left_side_step']), success_eps['left_side_step'])
    print(len(success_eps['right_side_step']), success_eps['right_side_step'])
    print(len(success_eps['move_behind']), success_eps['move_behind'])
    print(len(success_eps['move_until_visible']), success_eps['move_until_visible'])
