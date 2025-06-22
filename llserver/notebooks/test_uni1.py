import random
import time
from llserver.utils.handler import UniserverHandler

handler = UniserverHandler(port=8000)

info = handler.get_running_models()
model_ids = list(info["running_models"]["models"].keys())
info

model_id = model_ids[0]


task_number = 20
# # Make task_number requests to each model instance
to_recieve_tasks = []

total_start_time = time.time()
for i in range(task_number):
    prompt = "compare images"
    base_path = "/llserver/data/"
    image_paths = [
        base_path + "metatmp.png",
        base_path + "metatmp1.png",
    ]
    model_name_to_req = "gemini-pro-1.5"
    
    start_time = time.time()
    put_response = handler.put_task(model_id=model_id, prompt=prompt, image_paths=image_paths)
    task_id = put_response["task_id"]["task_id"]
    end_time = time.time()
    print(f"Time taken to put task: {end_time - start_time} seconds")
    
    start_time = time.time()
    while True:
        result = handler.get_task_result(model_id=model_id, task_id=task_id)
        # print(result)
        print(result["status"])
        if result["status"] == "in_progress" or result["status"] == "in queue":
            time.sleep(1)
        else:
            # print("="*30)
            # print(task_id)
            # print(result["result"])
            break
    end_time = time.time()
    print(f"Time taken to get task: {end_time - start_time} seconds")
    print("="*30)
    
    time.sleep(random.randint(3, 9))
    
total_end_time = time.time()
print(f"Total time taken: {total_end_time - total_start_time} seconds")
