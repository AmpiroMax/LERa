from transformers import RobertaTokenizer, T5ForConditionalGeneration
import string
import torch
from fiqa.language_processing.subtask import Subtask

exclude = set(string.punctuation)


class CodeT5:
    """
    CodeT5 seq-to-seq model for language instructions processing.
    There were trained 4 types of models:
        - "film" was trained on the same templates
        as in FILM (So Yeon Min et al.)
        - "no_recept" was trained on subtasks (obj, action)
        derived from ground true ALFRED trajectories.
        - "recept" was trained on (obj, optional receptacle, action)
        also derived from gt trajectories.
        - "recept+nav" was trained on "recept" subtasks enriched with
        navigational subtasks.
    Please, see the article for more details.
    """

    def __init__(self, device, instr_type='recept+nav'):

        self.tokenizer = RobertaTokenizer.from_pretrained(
            'Salesforce/codet5-base'
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            'Salesforce/codet5-base'
        )
        self.model.to(device)
        self.device = device
        # Model with receptacles prediction + navigation action (GotoLocation)
        if instr_type == 'recept+nav':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/recept+nav.bin', map_location=device)
            )
        # Receptacles only
        elif instr_type == 'recept':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/recept.bin', map_location=device)
            )
        # Model trained on FILM templates
        elif instr_type == 'film':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/film.bin', map_location=device)
            )
        # No receptacles, no navigation
        elif instr_type == 'no_recept':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/no_recept.bin', map_location=device)
            )
        else:
            assert False, f"Unknown instruction type: {instr_type}"

        self.instr_type = instr_type
        self.model.eval()

    def get_list_of_subtasks(self, traj_data):
        """

        Parameters
        ----------
        traj_data : object
             Trajectory data with NL instructions,
             obtained from ALFRED json-file for particular global task.

        Returns
        -------
        list_of_subtasks: list
             Output sequence of subtasks.

        """

        # Instructions preprocessing
        # (goal task concatenated with step-by-step instructions)
        r_idx = traj_data['repeat_idx']
        anns = traj_data['turk_annotations']['anns']
        goal = anns[r_idx]['task_desc'].lower().strip().replace('\n', '')
        goal = ''.join(ch for ch in goal if ch not in exclude)
        high_descs = [''.join(ch for ch in desc if ch not in exclude).lower().
                      strip().replace('\n', '') for desc in
                      anns[r_idx]['high_descs']]
        instructions = goal + ' . ' + ' . '.join(high_descs)

        input_ids = self.tokenizer(instructions, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, max_length=700)

        # Model output
        output = self.tokenizer.decode(generated_ids[0],
                                       skip_special_tokens=True)
        list_of_subtasks = [
            Subtask(tuple(t.split())) for t in output.split(' ; ')
        ]
        # print([(t.obj, t.recept, t.action) for t in list_of_subtasks])

        # TODO: task correctness assertion

        return list_of_subtasks
