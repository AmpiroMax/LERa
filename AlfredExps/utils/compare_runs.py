import argparse
import pickle
from glob import glob
from collections import defaultdict

from fiqa.language_processing import subtasks_helper


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
        '--before_changes_run_name_prefix', type=str,
        required=True,
        help='Prefix of the name of the directory inside results/[data_split]/'
            + ' where the logs are saved.'
    )
    parser.add_argument(
        '--before_changes_run_name_postfix', type=str, default='',
        help='Postfix of the name of the directory inside '
            + 'results/[data_split]/ where the logs are saved.'
    )
    parser.add_argument(
        '--after_changes_run_name_prefix', type=str,
        required=True,
        help='Prefix of the name of the directory inside results/[data_split]/'
            + ' where the logs are saved.'
    )
    parser.add_argument(
        '--after_changes_run_name_postfix', type=str, default='',
        help='Postfix of the name of the directory inside '
            + 'results/[data_split]/ where the logs are saved.'
    )

    return parser.parse_args()


def get_results(run_name_prefix, run_name_postfix):
    regexp = f'_[0-9]*_[0-9]*_' if run_name_postfix else f'_[0-9]*_[0-9]*'
    results_glob = glob(
        'results/' + args.split + '/' + run_name_prefix
        + regexp + run_name_postfix
    )
    if not results_glob:
        assert False, 'Logs with such names do not exist!'

    results = []
    for target_dir in sorted(results_glob):
        result = pickle.load(open(f'{target_dir}/stats_recs.p', 'rb'))
        results += result
    return results


if __name__ == '__main__':
    args = parse_args()
    if args.split in ['tests_seen', 'tests_unseen']:
        assert False, 'You are not allowed to analyse tests!'
    
    before_changes_results = get_results(
        args.before_changes_run_name_prefix, args.before_changes_run_name_postfix
    )
    after_changes_results = get_results(
        args.after_changes_run_name_prefix, args.after_changes_run_name_postfix
    )
    if len(before_changes_results) != len(after_changes_results):
        print(
            f'Warning! Lengths are not equal: {len(before_changes_results)} '
            + f'and {len(after_changes_results)}'
        )

    # Load scene names from the chosen data split
    scene_names = subtasks_helper.load_scene_names(
        args.split, 'alfred_utils/data/splits', 'alfred'
    )
    check_eps = defaultdict(list)
    print('These episodes have become unsuccessful:')
    for ep in range(min(len(before_changes_results), len(after_changes_results))):
        traj_data = subtasks_helper.load_traj(
            scene_names[ep], 'alfred_utils/data', 'alfred'
        )
        before_changes_success = before_changes_results[ep]['success']
        after_changes_success = after_changes_results[ep]['success']
        if before_changes_success and not after_changes_success:
            check_eps[traj_data['task_type']].append(ep)
    
    for key in sorted(check_eps.keys()):
        print(f'For {key}:', check_eps[key])
