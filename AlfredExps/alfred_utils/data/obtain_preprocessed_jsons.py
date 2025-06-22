# The script is based on 
# https://github.com/askforalfred/alfred/blob/master/models/train/train_seq2seq.py
import argparse
import json
from .preprocess import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Arguments for the preprocess script'
    )

    parser.add_argument(
        '--data', default='alfred_utils/data/json_2.1.0', help='Dataset folder'
    )
    parser.add_argument(
        '--splits', default='alfred_utils/data/splits/oct21.json',
        help='Json file containing train/dev/test splits'
    )
    parser.add_argument(
        '--pp_folder', default='pp', help='Folder name for preprocessed data'
    )
    # parser.add_argument(
    #     '--dout', default='exp/model:{model}', help='Where to save model'
    # )
    parser.add_argument(
        '--use_templated_goals', action='store_true',
        help='Use templated goals instead of human-annotated goal ' \
            + 'descriptions (only available for the train set)'
    )
    parser.add_argument(
        '--pframe', default=300, type=int,
        help='Image pixel size (assuming square shape eg: 300x300)'
    )
    parser.add_argument(
        '--fast_epoch', action='store_true', help='Fast epoch during debugging'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)

    # Preprocess and save
    print(
        'Preprocessing dataset and saving to %s folders... ' % args.pp_folder
        + 'This will take a while. Do this once as required.'
    )
    dataset = Dataset(args, None)
    dataset.preprocess_splits(splits)
