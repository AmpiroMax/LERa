"""
Functions for language instructions handling.
"""
import pickle
import os
import json
from typing import List, Optional, Dict

from fiqa.language_processing.subtask import Subtask

# Prepositions for receptacles to be used in questions generation
prepositions = {
    'armchair': 'on',
    'bed': 'on',
    'bowl': 'in',
    'box': 'in',
    'bathtubbasin': 'in',
    'cabinet': 'in',
    'coffeemachine': 'in',
    'coffeetable': 'on',
    'countertop': 'on',
    'desk': 'on',
    'diningtable': 'on',
    'drawer': 'in',
    'dresser': 'in',
    'fridge': 'in',
    'garbagecan': 'in',
    'handtowelholder': 'on',
    'laundryhamper': 'in',
    'microwave': 'in',
    'mug': 'in',
    'cup': 'in',
    'ottoman': 'on',
    'pan': 'on',
    'plate': 'on',
    'pot': 'in',
    'paintinghanger': 'on',
    'safe': 'in',
    'shelf': 'on',
    'sidetable': 'on',
    'sinkbasin': 'in',
    'sofa': 'on',
    'stoveburner': 'on',
    'tvstand': 'on',
    'toaster': 'in',
    'toilet': 'on',
    'toiletpaperhanger': 'on',
    'towelholder': 'on',
    'cart': 'in'
}


def load_traj(
    scene_name: dict, json_path: str,
    dataset: str, split: Optional[str] = None
) -> dict:
    if dataset == 'alfred':
        json_dir = json_path + '/json_2.1.0/' + scene_name['task'] + \
            '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
    elif dataset == 'teach':
        assert split is not None, 'The split must be provided!'
        json_dir = json_path + '/' + split + '/' + scene_name
    else:
        assert False,  f'Unknown dataset name: {dataset}'

    with open(json_dir, 'r') as f:
        traj_data = json.load(f)

    return traj_data

def write_processed_instructions_to_file(
    split: str, split_output: str, instr_type: str,
    dataset: str, path: str, gt: bool
):
    file_name = path + f'/{split}_instructions_processed_{instr_type}'
    if gt:
        file_name += '_gt'
    file_name += '_' + dataset
    with open(file_name + '.p', 'wb') as f:
        pickle.dump(split_output, f)


def load_processed_instructions(
    split: str, instr_type: str, 
    dataset: str, path: str, gt: bool
) -> Dict[tuple, Subtask]:
    file_name = path + f'/{split}_instructions_processed_{instr_type}'
    if gt:
        file_name += '_gt'
    file_name += '_' + dataset
    with open(file_name + '.p', 'rb') as f:
        instructions_processed = pickle.load(f)
    
    return instructions_processed

def get_gt_list_of_subtasks(
    split: str, instr_type: str,
    dataset: str, task_key: str, path: str
):
    file_name = path + f'{split}_{instr_type}_gt_{dataset}.p'
    with open(file_name, 'rb') as f:
        gt_lists = pickle.load(f)
    
    return [Subtask(s) for s in gt_lists[task_key]]


def load_scene_names(split: str, path: str, dataset: str) -> List[dict]:
    if dataset == 'alfred':
        with open(path + '/oct21.json', 'r') as f:
            scene_names = json.load(f)[split]
    elif dataset == 'teach':
        scene_names = list(filter(
        lambda x: '.json' in x, os.listdir(path=path + '/' + split)
        ))
    else:
        assert False,  f'Unknown dataset name: {dataset}'

    return scene_names


def generate_questions_from_subtask(
    subtask: Subtask, obj_in_hands: Optional[str] = None
) -> List[str]:
    """Generates questions for the `SubtaskChecker` class given a subtask.

    Parameters
    ----------
    subtask : Subtask
        Current subtask (obj, recept, action) that is to be transformed to
        the list of questions.
    obj_in_hands : str
        An object that is held in hands is required for PutObject action,
        if we use the subtasks without receptacles (obj, actions).

    Returns
    -------
    questions : list
        List of questions generated for the current task.
        NB: Currently, the size of the list is always 1.
    """
    obj, recept, action = subtask.obj, subtask.recept, subtask.action
    obj = obj.lower()
    # Transform 'potatosliced' --> 'sliced potato'
    if 'sliced' in obj:
        obj = 'sliced ' + obj.replace('sliced', '')
    if recept:
        recept = recept.lower()
    questions = []

    if action == 'PickupObject':
        # questions.append(f'Is the {obj} close enough to be picked up?')
        questions.append(f'Is the {obj} being held?')

    elif action == 'PutObject':
        if recept:
            questions.append(
                f'Is the {obj} {prepositions[recept]} the {recept}?')
        else:
            assert obj_in_hands is not None, \
                "action is 'PutObject', but obj_in_hands is None..."
            questions.append(
                f'Is the {obj_in_hands} {prepositions[obj]} the {obj}?')

    elif action in ('OpenObject', 'CloseObject'):
        questions.append(f'Is the {obj} opened?')

    elif action in ('ToggleObjectOn', 'ToggleObjectOff'):
        questions.append(f'Is the {obj} on?')

    elif action == 'SliceObject':
        questions.append(f'Is the {obj} sliced  ?')

    elif action == 'GotoLocation':
        questions.append(f'Is the {obj} visible?')

    else:
        assert False, f"Corrupted action {action}."

    return questions


def generate_existence_question(obj, output=None):
    """Generates an existence question for the object `obj`.

    Parameters
    ----------
    obj : str
        Current subtask's object.
    output : list of strings
        List of previous questions.

    Returns
    -------
    output : list of strings
        List of questions with existence question appended about the current
        object.
    """
    if output is None:
        output = []
    return output + [f'Is the {obj} visible?', ]


def add_navigation_subtasks(list_of_subtasks):
    """Inserts navigation subtasks manually for particular instruction types
    (recept, no_recept, film), since navigation is not predicted in these types,
    but it is needed for navigation module.

    Parameters
    ----------
    list_of_subtasks : list
        List of subtasks, obtained from CodeT5.

    Returns
    -------
    list_of_subtasks_with_navigation : list
        Output list with navigation subtasks inserted (obj, 'GotoLocation')
    """
    # Previous navigation object (None in the beginning of episode)
    prev_nav_obj = None
    list_of_subtasks_with_navigation = []
    obj_opened = False
    for i, subtask in enumerate(list_of_subtasks):
        obj, recept, action = subtask.obj, subtask.recept, subtask.action

        # Navigation object is either receptacle (if predicted),
        # or regular object. Since the subtask ('Floor', 'GotoLocation')
        # is meaningless, we omit it
        if recept is None or recept == 'Floor':
            # Avoid excessive navigation to objects,
            # contained inside some receptacle
            if i >= 1:
                if list_of_subtasks[i-1].action == 'OpenObject':
                    obj_opened = True
                if list_of_subtasks[i-1].action == 'CloseObject':
                    obj_opened = False
            # While the openable object is opened, navigation is not inserted
            if obj_opened:
                list_of_subtasks_with_navigation.append(subtask)
                continue
            # Avoid excessive navigation to faucet
            # and to the washed object after interaction with faucet
            if obj == 'Faucet' or \
            ((i >= 1) and list_of_subtasks[i-1].obj == 'Faucet'):
                list_of_subtasks_with_navigation.append(subtask)
                continue
            nav_obj = obj
        else:
            nav_obj = recept

        # Check whether we are already near the navigation object,
        # so we don't need to navigate to it
        if nav_obj != prev_nav_obj:
            list_of_subtasks_with_navigation.append(
                Subtask((nav_obj, 'GotoLocation'))
            )

        list_of_subtasks_with_navigation.append(subtask)
        # Save previous navigation object
        prev_nav_obj = nav_obj
    return list_of_subtasks_with_navigation
