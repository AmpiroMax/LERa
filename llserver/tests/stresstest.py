import requests
import time
from utils import put_task, get_task_result

# 1. Добавляем задачу в очередь
image_path = "~/work/meta_world/16_changed_state.png"
prompt = "Describe image"

tasks_ids = []
for i in range(1):
    task_response = put_task(image_path, prompt)
    task_id = task_response.get("task_id")
    tasks_ids.append(task_id)
    
for i in range(100):
    for task_id in tasks_ids:
        result_response = get_task_result(task_id)
        print(result_response)
