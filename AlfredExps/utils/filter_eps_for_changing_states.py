import json


if __name__ == '__main__':
    with open('alfred_utils/data/eps_for_changing_states.json', 'r') as f:
        data = json.load(f)
    data = data['valid_seen']

    for task_type in data.keys():
        data[task_type] = dict(filter(
            lambda kv: kv[0].split(', ')[1] == '0)', data[task_type].items()
        ))
        # Since the 417th episode can be completed sucessfully without interacting with
        # the fridge (a potato can be found outside it), also filter it:
        if task_type == 'pick_and_place_with_movable_recep':
            data[task_type] = dict(filter(
                lambda kv: kv[0].split(', ')[0] != "('trial_T20190910_001038_953470'",
                data[task_type].items()
            ))
    with open('alfred_utils/data/filtered_eps_for_changing_states.json', 'w') as f:
        json.dump({'valid_seen': data}, f)

    # Additionally filter "Faucet"
    data_without_faucet = dict()
    for task_type in data.keys():
        tmp_dict = dict(filter(
            lambda kv: 'Faucet' not in kv[1], data[task_type].items()
        ))
        if len(tmp_dict):
            data_without_faucet[task_type] = tmp_dict
    with open(
        'alfred_utils/data/filtered_eps_for_changing_states_without_Faucet.json', 'w'
    ) as f:
        json.dump({'valid_seen': data_without_faucet}, f)
