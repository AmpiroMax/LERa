import os
from fastapi import FastAPI, Form, Depends
from pydantic import BaseModel
import asyncio
from llserver.models.LlavaModel import LlavaModel  # Импорт вашего класса LlavaModel
from llserver.models.ECoTModel import ECoTModel  # Импорт вашего класса LlavaModel
from llserver.models.LLERa import LLERa  # Импорт вашего класса LlavaModel
from llserver.models.LLERa_with_API import LLERaAPI  # Импорт вашего класса LlavaModel
from contextlib import asynccontextmanager
from llserver.utils.custom_logger import CustomLogger  # Импортируем наш логгер
from typing import List

# Инициализация логгера
logger = CustomLogger(mode="logging")  # Можно переключить на "wandb"

# Модель для получения данных о задаче
class TaskResult(BaseModel):
    task_id: str

# Логика для инициализации и завершения работы через lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Логируем крупное сообщение при запуске приложения
    logger.log("=======================================")
    logger.log("========== ЗАПУСКАЕМ ПРОЦЕСС ==========")
    logger.log("=======================================")
    
    models = {
        "llava": LlavaModel,
        "ecot": ECoTModel,
        "llera": LLERa,
        "llera_api": LLERaAPI,
    }
    # Получаем название модели из переменной окружения или аргумента
    model_name = os.getenv("MODEL_NAME", "llava")  # По умолчанию LlavaModel


    if model_name not in models:
        raise ValueError(f"Указанная модель '{model_name}' не поддерживается. Доступные модели: {list(models.keys())}")

    # Логируем сообщение о начале инициализации модели
    logger.log(f"Инициализация модели {model_name} начинается...")

    # Создаем экземпляр LlavaModel
    model = models[model_name](logger=logger)  # Передаем логгер в класс модели

    # Запускаем фоновую задачу для обработки очереди задач
    worker_task = asyncio.create_task(model._task_worker())

    try:
        # Сохраняем модель в app.state
        app.state.model = model

        # Логируем сообщение об успешной инициализации модели
        logger.log(f"Модель {model_name} была успешно проинициализирована.")
        yield
    finally:
        # Завершение работы воркера при завершении сервера
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Завершение работы логгера (например, wandb)
        logger.shutdown()

# Инициализация FastAPI с новым lifespan
app = FastAPI(lifespan=lifespan)

# Получение модели из состояния приложения
def get_model():
    return app.state.model


@app.post("/put_task/")
async def put_task(
    image_paths: List[str] = [],
    prompt: str = Form(...),
    model: LlavaModel = Depends(get_model)
):
    """
    Ручка для добавления задачи с несколькими изображениями в очередь. Возвращает уникальный ID задачи.
    """
    task_id = await model.put_task(image_paths, prompt)
    logger.log(f"Задача с ID {task_id} поставлена в очередь")
    return {"task_id": task_id}

@app.post("/get_task_result/")
async def get_task_result(
    task: TaskResult,
    model: LlavaModel = Depends(get_model)
):
    """
    Ручка для получения результата выполнения задачи по ID.
    """
    result = await model.get_task_result(task.task_id)

    # Проверяем статус задачи и логируем его
    if result["status"] == "in queue":
        logger.log(f"Задача с ID {task.task_id} находится в очереди")
    elif result["status"] == "in progress":
        logger.log(f"Задача с ID {task.task_id} выполняется")
    elif result["status"] == "completed":
        logger.log(f"Задача с ID {task.task_id} завершена и удалена из очереди результатов")
    else:
        logger.log(f"Задача с ID {task.task_id} не найдена")

    return result