from transformers import RobertaTokenizer, T5ForConditionalGeneration
import string
import torch
from fiqa.language_processing.subtask import Subtask

exclude = set(string.punctuation)


class CodeT5Teach:
    """
    CodeT5 seq-to-seq model for TEACh dialogs processing.
    "no_recept" model was trained on subtasks (obj, action)
    derived from Follower's ground true TEACh trajectories.
    "film" model was trained on manually created templates for each
    TEACh task type.
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
        # No receptacles, no navigation
        if instr_type == 'no_recept':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/teach_no_recept.bin', map_location=device)
            )
        elif instr_type == 'film':
            self.model.load_state_dict(torch.load(
                'fiqa/checkpoints/teach_film.bin', map_location=device)
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
             obtained from TEACh json-file for particular global task.

        Returns
        -------
        list_of_subtasks: list
             Output sequence of subtasks.

        """

        # Dialogs preprocessing
        dialogs = ' '.join(
            ': '.join(utt) for utt in traj_data['dialog_history_cleaned']
            )
        input_ids = self.tokenizer(dialogs, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, max_length=700)

        # Model output
        output = self.tokenizer.decode(generated_ids[0],
                                       skip_special_tokens=True)
        list_of_subtasks = [
            Subtask(tuple(t.split())) for t in output.split(' ; ')
        ]

        # TODO: task correctness assertion

        return list_of_subtasks
