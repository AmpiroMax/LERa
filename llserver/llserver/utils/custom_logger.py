import logging
import inspect
import wandb
import os

class CustomLogger:
    def __init__(self, mode="logging", log_file="app.log"):
        """
        Инициализация логгера.
        :param mode: 'logging' или 'wandb'
        :param log_file: Путь к лог-файлу.
        """
        self.mode = mode
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.INFO)
        
        # Настройка логгера для записи в файл
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Настройка wandb, если выбран режим 'wandb'
        if self.mode == "wandb":
            wandb.init(project="llava-task-logger", name="LlavaModelTaskLog")

    def log(self, message):
        """
        Логирование сообщений с информацией о классе и методе.
        """
        # Получаем текущий фрейм вызова
        frame = inspect.stack()[1]
        method_name = frame.function

        # Получаем информацию о классе
        cls = frame.frame.f_locals.get('self', None)
        class_name = cls.__class__.__name__ if cls else 'Global'

        # Формируем полное сообщение
        full_message = f"[{class_name}.{method_name}] {message}"

        # Логирование в файл и стандартный вывод
        if self.mode == "logging":
            self.logger.info(full_message)
        elif self.mode == "wandb":
            wandb.log({"message": full_message})
            self.logger.info(full_message)

    def shutdown(self):
        """
        Завершение работы, если используется wandb.
        """
        if self.mode == "wandb":
            wandb.finish()