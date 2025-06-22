import uuid
import docker
from fastapi import FastAPI, Form, Body
from pydantic import BaseModel
from typing import List, Dict

from llserver.utils.custom_logger import CustomLogger 
from llserver.utils.utils import (
    put_task as utils_put_task,
    get_task_result as utils_get_task_result,
    read_json,
    write_json
)

class ModelInfo(BaseModel):
    model_name: str
    model_id: str
    container_id: str
    port: int

class RunningModels(BaseModel):
    models: Dict[str, ModelInfo]

base_path = '/home/mpatratskiy/work/meta_world/llserver'

running_models_data = read_json(base_path+'/llserver/server/running_models.json')
running_models_info: RunningModels = RunningModels(models={
    model_id: ModelInfo(**model_info) 
    for model_id, model_info in running_models_data["models"].items()
})

docker_client = docker.from_env()
logger = CustomLogger(mode="logging") 


app = FastAPI()


@app.get("/")
async def ping():
    """
    Ping the server to check if it is working.
    """
    return {"status": "Server is working"}


@app.get("/running_models/")
async def get_running_models():
    """
    Ручка для получения списка запущенных моделей.
    """
    return {"running_models": running_models_info}


@app.post("/start_model/")
async def start_model(model_name: str):
    used_ports = [model.port for model in running_models_info.models.values()]
    used_ports.append(8080)
    free_port = max(used_ports) + 1
    container = None
    try:
        # Запуск контейнера
        model_id = str(uuid.uuid4())
        container = docker_client.containers.run(
            image=f"llmserver.{model_name}",
            name=f"llmserver.{model_name}.{model_id}",
            ports={'8080/tcp': free_port},
            volumes={
                base_path+'/data': {
                    'bind': '/llserver/data',
                    'mode': 'rw',
                },
                base_path+'/logs': {
                    'bind': '/llserver/logs',   
                    'mode': 'rw',
                },
                base_path+'/models': {
                    'bind': '/home/models',
                    'mode': 'rw',
                },
                '/home/mpatratskiy/work/eai_fiqa/data': {
                    'bind': '/llserver/alfred/data',
                    'mode': 'rw',
                },
            },
            detach=True,
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,  # Use all available GPUs
                    capabilities=[['gpu']]
                )
            ]

        )
        
        logger.log(f"Модель {model_name} c id {model_id} запущена в контейнере {container.id}")
        running_models_info.models[model_id] = ModelInfo(model_name=model_name, port=free_port, model_id=model_id, container_id=container.id)
        running_models_info_json = running_models_info.dict()
        write_json(running_models_info_json, base_path+'/llserver/server/running_models.json')
        return {
            "model_name": model_name,
            "status": "Model started",
            "container_id": container.id,
            "port": free_port,
            "model_id": model_id,
        }
    
    except Exception as e:
        
        if container is not None:
            # Log the container that was created but encountered an error and is being stopped
            logger.log(f"Container {container.id} was created, but an error occurred, and it is being stopped")
            container.stop()
            container.remove()
            
        logger.log(f"Ошибка при запуске модели {model_name}: {str(e)}")
        return {
            "model_name": model_name,
            "status": "Error in starting model",
            "error": str(e),
            "container_id": None,
            "port": None,
            "model_id": None,
        }
        

@app.post("/stop_model/")
async def stop_model(model_id: str):
    """
    Ручка для остановки конкретной запущенной модели.
    """
    if model_id not in running_models_info.models:
        return {
            "model_id": model_id,
            "status": "Model not running",
            "container_id": None,
            "port": None,
            "model_id": None,
        }
    
    try:
        model_name = running_models_info.models[model_id].model_name
        model_port = running_models_info.models[model_id].port
        container = docker_client.containers.get(f"llmserver.{model_name}.{model_id}")
        container.stop()
        container.remove()
        logger.log(f"Модель {model_id} остановлена и контейнер {container.id} удален")
        del running_models_info.models[model_id]
        running_models_info_json = running_models_info.dict()
        write_json(running_models_info_json, base_path+'/llserver/server/running_models.json')
        return {
            "model_id": model_id,
            "status": "Model stopped",
            "container_id": container.id,
            "port": model_port,
        }
    
    except Exception as e:
        logger.log(f"Ошибка при остановке модели {model_id}: {str(e)}")
        return {
            "model_id": model_id,
            "status": "Error in stopping model",
            "error": str(e),
            "container_id": None,
            "port": None,
        }


@app.post("/put_task/")
async def put_task(
    model_id: str = Body(...),
    prompt: str = Body(...),
    image_paths: List[str] = Body(default=[]),
    extra_params: dict[str, str] = Body(default={})
):
    logger.log(f"extra_params: {extra_params}")
    port = running_models_info.models[model_id].port
    task_id = utils_put_task(image_paths, prompt, port, **extra_params)
    logger.log(f"Задача с ID {task_id} поставлена в очередь модели {model_id}")
    return {"task_id": task_id}



@app.post("/get_task_result/")
async def get_task_result(
    model_id: str,
    task_id: str
):
    """
    Ручка для получения результата выполнения задачи по ID.
    """
    port = running_models_info.models[model_id].port
    result = utils_get_task_result(task_id, port)

    # Проверяем статус задачи и логируем его
    if result["status"] == "in queue":
        logger.log(f"Задача с ID {task_id} находится в очереди")
    elif result["status"] == "in progress":
        logger.log(f"Задача с ID {task_id} выполняется")
    elif result["status"] == "completed":
        logger.log(f"Задача с ID {task_id} завершена и удалена из очереди результатов")
    else:
        logger.log(f"Задача с ID {task_id} не найдена")

    return result