from collections import defaultdict
import json
import os

from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    dataset_folder = 'fiqa/language_processing/dataset_for_pose_correction'
    meta_folder = 'metas'
    pose_correction_types = [
        'left_side_step', 'right_side_step', 'move_behind', 'move_until_visible'
    ]
    pose_corr_type2task_type2pose_corrs = defaultdict(lambda: defaultdict(list))
    for pose_correction_type in pose_correction_types:
        pose_corr_metas_folder = os.path.join(
            dataset_folder, pose_correction_type, meta_folder
        )
        for entry in sorted(map(lambda x: x.path, os.scandir(pose_corr_metas_folder))):
            with open(entry, 'r') as f:
                pose_corr = json.load(f)
            pose_corr_type2task_type2pose_corrs[pose_correction_type][
                pose_corr['task_type']
            ].append(pose_corr['rgb_idx'])

    task_types = [
        'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep',
        'pick_clean_then_place_in_recep', 'pick_and_place_with_movable_recep',
        'look_at_obj_in_light', 'pick_and_place_simple', 'pick_two_obj_and_place'
    ]
    pose_corr_type2corrs_cnts = dict()
    for pose_correction_type in pose_correction_types:
        task_type2cnts = {task_type: 0 for task_type in task_types}  # Special ordering
        for task_type in pose_corr_type2task_type2pose_corrs[pose_correction_type].keys():
            task_type2cnts[task_type] = len(
                pose_corr_type2task_type2pose_corrs[pose_correction_type][task_type]
            )
        pose_corr_type2corrs_cnts[pose_correction_type] = np.array(
            list(task_type2cnts.values()), dtype=int
        )

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_title('Pose corrections distribution by task type')
    task_types = list(map(lambda x: x[:20], task_types))
    bottom = np.zeros(len(task_types), dtype=int)
    for pose_correction, corrs_cnt in pose_corr_type2corrs_cnts.items():
        ax.bar(task_types, corrs_cnt, label=pose_correction, bottom=bottom)
        bottom += np.array(corrs_cnt)
    ax.legend()
    fig.savefig('utils/data/task_type2cnt_by_pose_correction_type')
    plt.close(fig)
