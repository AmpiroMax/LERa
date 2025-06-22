import argparse
import pickle
from glob import glob
from tabulate import tabulate
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyzer of runs stats')
    parser.add_argument(
        '--split', type=str,
        choices=[
            'train', 'valid_unseen', 'valid_seen', 'tests_seen', 'tests_unseen'
        ],
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
        '--run_name_postfix', type=str, default='',
        help='Postfix of the name of the directory inside '
            + 'results/[data_split]/ where the logs are saved.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.split in ['tests_seen', 'tests_unseen']:
        assert False, 'You are not allowed to analyse tests!'

    regexp = f'_[0-9]*_[0-9]*_' if args.run_name_postfix else f'_[0-9]*_[0-9]*'
    results_glob = glob(
        'results/' + args.split + '/' + args.run_name_prefix
        + regexp + args.run_name_postfix
    )
    if not results_glob:
        assert False, 'Logs with such names do not exist!'

    results = []
    for target_dir in sorted(results_glob):
        result = pickle.load(open(f'{target_dir}/stats_recs.p', 'rb'))
        results += result

    # Number of succeeded episodes
    successes = sum(s['success'] for s in results) * 100
    # GC success
    gcs = sum(s['log_entry']['goal_condition_success'] for s in results) * 100
    
    # Success SPL
    s_spl = sum(s['log_entry']['success_spl'] for s in results)
    # GCS SPL
    pc_spl = sum(s['log_entry']['goal_condition_spl'] for s in results)
    # Path length weighted SPL metrics 
    plw_s_spl = sum(
        s['log_entry']['path_len_weighted_success_spl'] for s in results
    )
    plw_gc_spl = sum(
        s['log_entry']['path_len_weighted_goal_condition_spl'] for s in results
    )
    # Navigation module success
    total_nav_success = [0, 0]
    for episode_info in results:
        total_nav_success[0] += episode_info['log_entry']['nav_success'][0]
        total_nav_success[1] += episode_info['log_entry']['nav_success'][1]
    total_nav_success = total_nav_success[0] / total_nav_success[1]

    # Task type metrics
    type_dict = {
        'pick_and_place_simple': {'success': [], 'gcs' : []},
        'pick_two_obj_and_place': {'success': [], 'gcs' : []},
        'pick_cool_then_place_in_recep': {'success': [], 'gcs' : []},
        'pick_clean_then_place_in_recep': {'success': [], 'gcs' : []},
        'pick_heat_then_place_in_recep': {'success': [], 'gcs' : []},
        'pick_and_place_with_movable_recep': {'success': [], 'gcs' : []},
        'look_at_obj_in_light' : {'success': [], 'gcs' : []}
    }
    for s in results:
        t_succ = s['success']
        t_gcs = s['log_entry']['goal_condition_success']
        type_dict[s['log_entry']['type']]['success'].append(t_succ)
        type_dict[s['log_entry']['type']]['gcs'].append(t_gcs)

    type_table = [
        ['Task Type', 'Num episodes', 'Num success episodes', 'Success', 'GCS']
    ]
    for task_type in type_dict.keys():
        type_succs = type_dict[task_type]['success']
        type_gcss = type_dict[task_type]['gcs']
        type_table.append(
            [task_type, f'{len(type_succs)}', f'{sum(type_succs)}', 
            f'{sum(type_succs) / len(type_succs) if len(type_succs) > 0 else 0.: .4f}', 
            f'{sum(type_gcss) / len(type_gcss) if len(type_gcss) > 0 else 0.: .4f}']
        )

    print(tabulate(type_table))

    print(f"Number of episodes: {len(results)}")
    print(f"Number of success episodes: {successes}")
    print(f"Success SPL: {s_spl / len(results): .4f}")
    print(f"GCS SPL: {pc_spl / len(results): .4f}")
    print(f"PLW Success SPL: {plw_s_spl / len(results): .4f}")
    print(f"PLW GCS SPL: {plw_gc_spl / len(results): .4f}")
    print(f"Nav module Success: {total_nav_success: .4f}")
