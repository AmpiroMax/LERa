from re import L
from PIL import Image
import numpy as np
import torch
from typing import Union


class Logger:
    def __init__(self):
        self.log_dir = None
        self.img_dir = None
        self.log_file = None
        self.img_loc = None

        self.time_length = 28
        self.subtask_length = 50
        self.action_length = 16
        self.step_length = 6
        self.sucess_length = 34
        self.lengths = [
            self.time_length, self.subtask_length, self.action_length, 
            self.step_length, self.sucess_length
        ]

    def set_log_dir(self, log_dir: str):
        self.log_dir = log_dir
        self.img_dir = log_dir + '/images'

    def reset(self, episode_idx: int):
        self.log_file = f'{episode_idx}.txt'
        self.img_loc = self.img_dir + f'/{episode_idx}'
        with open(self.log_dir + f'/{self.log_file}', 'a') as f:
            f.write(
                ''.join([
                    'Time' + ' ' * (self.time_length - 4), 
                    'Subtask' + ' ' * (self.subtask_length - 7), 
                    'Action' + ' ' * (self.action_length - 6), 
                    'S#' + ' ' * (self.step_length - 2), 
                    'Success_(GT_or_Estim_or_Checker)  ', 
                    'Error'
                ]) + '\n' + '=' * (sum(self.lengths) + len('Error')) + '\n'
            )

    def save_img(self, img: Union[np.ndarray, torch.Tensor], img_name: str):
        img = Image.fromarray(np.uint8(img))
        img.save(f'{self.img_loc}/{img_name}.png')

    def log(self, log_entity: dict):
        # Align columns (order is utilized!)
        for i, key in enumerate(log_entity.keys()):
            if key == 'error':
                break
            log_entity[key] = str(log_entity[key]) \
                + ' ' * (self.lengths[i] - len(str(log_entity[key])))
        
        with open(self.log_dir + f'/{self.log_file}', 'a') as f:
            f.write(
                ''.join(str(value) for value in log_entity.values()) + '\n'
            )

    def log_warning(self, msg: str):
        with open(self.log_dir + f'/{self.log_file}', 'a') as f:
            f.write(f'Warning! {msg}' + '\n')

    def log_error(self, msg: str):
        with open(self.log_dir + f'/{self.log_file}', 'a') as f:
            f.write(f'Error! {msg}' + '\n')

    def log_msg(self, msg: str):
        with open(self.log_dir + f'/{self.log_file}', 'a') as f:
            f.write(msg + '\n')


logger = Logger()
