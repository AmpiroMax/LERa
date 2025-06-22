# import argparse
from collections import defaultdict
import json
from matplotlib import pyplot as plt
import numpy as np


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description='Analyzer of state changes')
#     parser.add_argument(
#         '--split', type=str, choices=['valid_seen', 'valid_unseen'], required=True,
#         help='ALFRED data split'
#     )
#     return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    with open(
        f'alfred_utils/data/filtered_valid_seen_eps_for_changing_states.json', 'r'
    ) as f:
        valid_seen = json.load(f)["valid_seen"]
    with open(
        f'alfred_utils/data/filtered_valid_unseen_eps_for_changing_states.json', 'r'
    ) as f:
        valid_unseen = json.load(f)["valid_unseen"]

    ordered_task_types = [
        'pick_and_place_simple',
        'pick_two_obj_and_place',
        'pick_cool_then_place_in_recep',
        'pick_clean_then_place_in_recep',
        'pick_heat_then_place_in_recep',
        'pick_and_place_with_movable_recep',
        'look_at_obj_in_light',
    ]
    obj2task_types = defaultdict(
        lambda: {task_type: 0 for task_type in ordered_task_types}
    )
    for data in [valid_seen, valid_unseen]:
        for task_type in data.keys():
            for task in data[task_type].keys():
                for obj in data[task_type][task]:
                    obj2task_types[obj][task_type] += 1

    ordered_objs = {"Fridge", "Microwave", "FloorLamp", "DeskLamp"}
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.06)
    ax = plt.subplot(1, 1, 1)
    ax.set_title('Objects distribution by task type')
    ax.title.set_fontsize(28)
    bottom = np.zeros(len(ordered_task_types), dtype=int)
    for obj in ordered_objs:
        task_type_cnts = obj2task_types[obj]
        ax.bar(task_type_cnts.keys(), task_type_cnts.values(), label=obj, bottom=bottom)
        bottom += np.array(list(task_type_cnts.values()))
    ax.legend(fontsize=17)
    ax.set_xticklabels(list(range(len(ordered_task_types))))
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(16)
    ax.set_ylim(top=100)
    fig.savefig(f'utils/data/filtered_valid_seen+unseen_task_types2obj_cnts.png')
    plt.close(fig)
