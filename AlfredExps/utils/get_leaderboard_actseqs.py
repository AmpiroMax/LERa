import argparse
import json


parser = argparse.ArgumentParser(description='Leaderboard preparation options')
parser.add_argument(
    '--test_split', type=str,
    choices=['seen', 'unseen', 'seen_unseen'],
    required=True,
    help='Which test split to prepare: seen, unseen or both.'
)
parser.add_argument(
    '--run_name', type=str,
    required=True,
    help='The name of the directory inside results/[data_split]/ ' +
         'where images are saved.'
)
parser.add_argument(
    '--debug', action='store_true', default=False, 
    help='If true, the lengths of the results are not checked.'
)

args = parser.parse_args()

seen_results = {'tests_seen': [], 'tests_unseen': []}
unseen_results = {'tests_seen': [], 'tests_unseen': []}
if args.test_split in ['seen', 'seen_unseen']:
    with open(f'results/tests_seen/{args.run_name}/for_leaderboard.json') as f:
        seen_results = json.load(f)
    if not args.debug:
        assert len(seen_results['tests_seen']) == 1533, 'Incorrect length!'
if args.test_split in ['unseen', 'seen_unseen']:
    with open(
        f'results/tests_unseen/{args.run_name}/for_leaderboard.json'
    ) as f:
        unseen_results = json.load(f)
    if not args.debug:
        assert len(unseen_results['tests_unseen']) == 1529, 'Incorrect length!'

total_results = {
    'tests_seen': seen_results['tests_seen'], 
    'tests_unseen': unseen_results['tests_unseen']
}
with open(f'results/{args.run_name}_for_leaderboard.json', 'w') as r:
    json.dump(total_results, r, indent=4, sort_keys=True)
print(f'Saved into results/{args.run_name}_for_leaderboard.json!')
