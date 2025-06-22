import requests
import hashlib
import json

# Unified get / put functions

def put_task(image_paths, prompt, port, **kwargs):
    """
    Функция для добавления задачи с несколькими изображениями в очередь.
    """
    url = f"http://127.0.0.1:{port}/put_task/"
    data = {
        'prompt': prompt,
        'image_paths': image_paths,
        'extra_params': kwargs
    }
    
    response = requests.post(url, json=data)
    return response.json()


def get_task_result(task_id, port):
    """
    Функция для получения результата задачи по её ID.
    """
    url = f"http://127.0.0.1:{port}/get_task_result/"
    params = {
        "task_id": task_id
    }
    
    response = requests.post(url, params=params)
    return response.json()


def get_string_hash(input_string):
    """
    Function to calculate the MD5 hash of a given string.
    """
    return hashlib.md5(input_string.encode()).hexdigest()


def read_json(path: str) -> dict:
    """
    Function to read data from a JSON file.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data: dict, path: str):
    """
    Function to write data to a JSON file.
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
