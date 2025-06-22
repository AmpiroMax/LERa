import os
from glob import glob
import argparse
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyzer of runs stats')
    parser.add_argument(
        '--split', type=str,
        choices=['valid_unseen', 'valid_seen', 'tests_seen', 'tests_unseen'],
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

    debug_eps = []
    nav_stats = Counter()
    seg_checker_stats = Counter()
    seg_stats = Counter()
    interact_fails = Counter()
    checker_stats = Counter()
    total_episodes = 0
    for target_dir in sorted(results_glob):
        for entry in os.scandir(target_dir):
            if not(entry.is_file() and entry.name[-4:] == '.txt'):
                continue
            total_episodes += 1
            with open(entry.path, 'r') as f:
                logs = f.readlines()[2:]
            # If you want to find episodes with specific conditions, 
            # uncomment this:
            # with open(entry.path, 'r') as f:
            #     content = f.read()
            #     if 'KeyChain' in content and 'False' in content:
            #         debug_eps.append(entry.name)
            #     else:
            #         continue

            last_subtask_type = ''
            last_action_success = ''
            for j in range(0, len(logs) - 1):
                if (
                    'Warning' in logs[j] or 'Error' in logs[j] 
                    or logs[j][0] == ' '
                ):
                    if 'Oracle nav does not have a valid position' in logs[j]:
                        print('Oracle nav did not find a pos in', entry.name)
                    continue
                splitted = logs[j].split()
                subtask_type = splitted[2][1:-1]
                subtask_target = splitted[3][:-1]
                action = splitted[5]
                steps_taken = splitted[6]
                success = splitted[7]

                if subtask_type == 'GotoLocation':
                    if action == 'StopNav':  # Navigator
                        nav_stats[subtask_target + ':' + success[3:]] += 1
                    elif action == 'GotoLocation':  # Checker
                        checker_success = last_action_success[3:] == success[8:]
                        seg_checker_stats[subtask_target + ':' + str(checker_success)] += 1
                else:
                    if 'GT' in success:  # Interaction
                        seg_stats[subtask_target + ':' + success[3:]] += 1  # TODO: add nav:True check
                        if 'False' in success:
                            interact_fails[action] += 1
                    elif 'Checker' in success:  # Checker
                        checker_success = last_action_success[3:] == success[8:]
                        checker_stats[subtask_target + ':' + str(checker_success)] += 1

                last_action_success = success
                last_subtask_type = subtask_type

    if len(debug_eps) > 0:
        print('Episodes to analyze:', *sorted(debug_eps))
    print(f'Total episodes: {total_episodes}')
    print('Interaction fails per action type:', list(interact_fails.items()))
    all_stats = [nav_stats, seg_checker_stats, seg_stats, checker_stats]
    all_names = ['Nav', 'Seg existence', 'Seg interaction', 'Checker']
    for stats, phase in zip(all_stats, all_names):
        print(f'{phase} stats:')
        accuracy_per_obj = {}
        for k, v in stats.items():
            if 'True' in k:
                fails = stats.get(k[:-4] + 'False', 0)
                accuracy_per_obj[k[:-5]] = (v / (v + fails), v + fails)
            elif 'False' in k:
                succs = stats.get(k[:-5] + 'True', 0)
                accuracy_per_obj[k[:-6]] = (succs / (succs + v), succs + v)
        for item in sorted(accuracy_per_obj.items(), key=lambda x1: x1[0]):
            print(f'({item[0]}, {item[1][0]: .3f}, {item[1][1]}), ', end='')
        print('\n', end='')
