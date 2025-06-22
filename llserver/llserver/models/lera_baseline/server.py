import uvicorn
from fastapi import FastAPI, Form, Depends, Body
import asyncio
from LLERa_Baseline import LLERa_Baseline
from contextlib import asynccontextmanager
from llserver.utils.custom_logger import CustomLogger
from typing import List


# Инициализация логгера
logger = CustomLogger(mode="logging", log_file="/llserver/logs/lera_baseline.log")  # Можно переключить на "wandb"


# Логика для инициализации и завершения работы через lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Логируем крупное сообщение при запуске приложения
    logger.log("=======================================")
    logger.log("========== ЗАПУСКАЕМ ПРОЦЕСС ==========")
    logger.log("=======================================")
    
    # Логируем сообщение о начале инициализации модели
    logger.log(f"Инициализация модели LLERaBaseline начинается...")

    model = LLERa_Baseline(logger=logger)  # Передаем логгер в класс модели

    # Запускаем фоновую задачу для обработки очереди задач
    worker_task = asyncio.create_task(model._task_worker())

    try:
        # Сохраняем модель в app.state
        app.state.model = model

        # Логируем сообщение об успешной инициализации модели
        logger.log(f"Модель LLERaBaseline была успешно проинициализирована.")
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
    prompt: str = Body(...),
    image_paths: List[str] = Body(default=[]),
    extra_params: dict = Body(default={}),
    model: LLERa_Baseline = Depends(get_model),
):
    """
    Ручка для добавления задачи с несколькими изображениями в очередь. Возвращает уникальный ID задачи.
    """
    logger.log(f"extra_params: {extra_params}")
    task_id = await model.put_task(image_paths, prompt, **extra_params)
    logger.log(f"Задача с ID {task_id} поставлена в очередь")
    return {"task_id": task_id}


@app.post("/get_task_result/")
async def get_task_result(
    task_id: str,
    model: LLERa_Baseline = Depends(get_model)
):
    """
    Ручка для получения результата выполнения задачи по ID.
    """
    result = await model.get_task_result(task_id)

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)