"""
Script for creating output file with preprocessed instructions depending on
chosen data split.
"""

import argparse
import torch
from tqdm import tqdm
from fiqa.language_processing.model import CodeT5
from fiqa.language_processing.model_teach import CodeT5Teach
from fiqa.language_processing import subtasks_helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='FIQA instruction preprocessing'
    )

    parser.add_argument(
        '--split', type=str,
        choices=['valid_unseen', 'valid_seen', 'tests_seen', 'tests_unseen'],
        required=True,
        help='data split'
    )

    parser.add_argument(
        '--instr_type', type=str, default='recept+nav',
        choices=['film', 'no_recept', 'recept', 'recept+nav'],
        help='Instructions processing type. ' +
             'See fiqa/language_processing/model.py'
    )

    parser.add_argument(
        '--output_dir', type=str,
        default='fiqa/language_processing/processed_instructions',
        help='Path where to save the output'
    )

    parser.add_argument(
        '--dataset',
        default='alfred',
        choices=['alfred', 'teach'],
        help='Dataset to process'
    )

    parser.add_argument(
        '--gt', action='store_true',
        help='Ground truth plans'
    )

    parser.add_argument(
        '--eval', action='store_true',
        help='Evaluate model'
    )

    args = parser.parse_args()
    return args

def process_instructions(args, device):
    """A function for ALFRED or TEACh instructions processing."""
    split_output = {}
    if args.dataset == 'alfred':
        scene_names_path =  'alfred_utils/data/splits'
        json_path = 'alfred_utils/data'
        if not args.gt:
            model = CodeT5(device, args.instr_type)
    elif args.dataset == 'teach':
        scene_names_path = 'teach_utils/data/tfd_instances'
        json_path = 'teach_utils/data/tfd_instances'
        if not args.gt:
            model = CodeT5Teach(device, args.instr_type)
    else:
        assert False,  f'Unknown dataset name: {args.dataset}'
    
    scene_names = subtasks_helper.load_scene_names(
        args.split, scene_names_path, args.dataset
    )

    if args.eval:
        accuracy = 0
    for name in tqdm(scene_names):
        traj_data = subtasks_helper.load_traj(
            name, json_path, args.dataset, args.split)
        if args.dataset == 'alfred':
            # Task is identified by its task_id and repeat_index for ALFRED
            task_key = (traj_data['task_id'], traj_data['repeat_idx'])
        else:
            # Task is identified by its game_id for TEACh
            task_key = traj_data['game_id']
        if not args.gt:
            list_of_subtasks = model.get_list_of_subtasks(traj_data)
        else:
            list_of_subtasks = subtasks_helper.get_gt_list_of_subtasks(
                args.split, args.instr_type, args.dataset, task_key, 
                'fiqa/language_processing/processed_instructions/'
            )
        if args.eval:
            gt_subtasks = subtasks_helper.get_gt_list_of_subtasks(
                args.split, args.instr_type, args.dataset, task_key, 
                'fiqa/language_processing/processed_instructions/')
            if list_of_subtasks == gt_subtasks:
                accuracy += 1

        # Insert navigation subtasks for particular instruction types manually,
        # when navigation is not predicted
        if args.instr_type in ('film', 'no_recept', 'recept'):
            list_of_subtasks = subtasks_helper.add_navigation_subtasks(
                list_of_subtasks
            )
        split_output[task_key] = list_of_subtasks
        # print([(t.obj, t.recept, t.action) for t in split_output[task_key]])

    if args.eval:
        print(f"Accuracy: {accuracy / len(scene_names) * 100:.2f}")
    subtasks_helper.write_processed_instructions_to_file(args.split,
                                                         split_output,
                                                         args.instr_type,
                                                         args.dataset,
                                                         args.output_dir,
                                                         args.gt)

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    process_instructions(args, device)


if __name__ == '__main__':
    arguments = parse_args()
    print('Arguments:', arguments)
    main(arguments)
