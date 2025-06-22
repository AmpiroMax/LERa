import asyncio
import uuid
import warnings
from llserver.prompts.prompts_baseline import PROMPTS as baseline_prompts

warnings.filterwarnings("ignore")

from openai import OpenAI
import base64


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class LLERa_Baseline:
    def __init__(self, logger=None):
        self.logger = logger
        self.logger.log(f"Начинаем приветствие...")
        # https://aitunnel.ru/#models
        self.client = OpenAI(
            api_key="sk-aitunnel-lRVzfpdyQrVrYJGyxiFkgMMXLSz5P2Oz", # Ключ из нашего сервиса
            base_url="https://api.aitunnel.ru/v1/",
        )
        
        self.task_queue = asyncio.Queue()  # Очередь задач
        self.result_queue = {}  # Результаты задач
        self.task_status = {}  # Статусы задач
        self.logger.log(f"Привет, я LLERa_Baseline")


    def _predict(self, prompt, image_paths=[], model="gpt-4o"):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        self.logger.log(f"image_paths: {image_paths}")
        for image_path in image_paths:
            self.logger.log("converting image to base64")
            try:
                base64_image = encode_image(image_path)
            except Exception as e:
                print(e)
                self.logger.log(f"error: {e}")
                raise e
            self.logger.log(f"base64_image_length: {len(base64_image)}")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
        
        self.logger.log(f"Predicting with content length={len(content)} and model name={model}")
        
        completion = self.client.chat.completions.create(
            messages=messages,
            max_tokens=1000,
            model=model
        )

        text_outputs = completion.choices[0].message.content
        return text_outputs

    async def _process_task(self, task_id, image_paths, prompt, **kwargs):
        """
        Асинхронная обработка одной задачи с несколькими изображениями.
        """
        # Обновляем статус задачи до "in progress"
        self.task_status[task_id] = "in progress"
        self.logger.log(f"Задача с ID {task_id} начала выполняться")
        self.logger.log(f"kwargs: {kwargs}")
        
        def replace_template_with_info(template, info):
            for name, value in info.items():
                template = template.replace(name, value)
            return template
        

        self.logger.log("="*60)
        # prompt = "[goal]###[current_plan]###[available_actions]"
        goal, success_actions, current_plan, available_actions = prompt.split("###")
        first_action = current_plan.split(",")[0]
        info = {
            "[goal]": goal,
            "[success_actions]": success_actions,
            "[current_plan]": current_plan,
            "[available_actions]": available_actions,
            "[first_action]": first_action,
        }

        model_name = kwargs.get("model", "gpt-4o")

        prompt_templates = baseline_prompts

        if "alfred" in model_name:
            template_name = model_name.split("###")[1]
            prompt_templates = prompt_templates[template_name]
            model_name = model_name.split("###")[0]
            self.logger.log("Using alfred variant of prompts")

        self.logger.log(f"goal: {goal}")
        self.logger.log(f"success_actions: {success_actions}")
        self.logger.log(f"current_plan: {current_plan}")   
        self.logger.log(f"available_actions: {available_actions}")
        self.logger.log(f"image number: {len(image_paths)}")
        self.logger.log(f"model to predict: {model_name}")
        self.logger.log("="*30)

        # Replan
        replan_request = replace_template_with_info(prompt_templates["replan"], info)
        self.logger.log(f"replan_request: \n{replan_request}\n")
        replan_response = self._predict(replan_request, image_paths, model=model_name)
        info["[replan_response]"] = replan_response
        self.logger.log(f"replan_response: \n{replan_response}\n")
        self.logger.log("="*60)

        # Обновляем результат и статус задачи
        self.result_queue[task_id] = info

        self.task_status[task_id] = "completed"
        self.logger.log(f"Задача {task_id} успешно завершена")

    async def _task_worker(self):
        """
        Асинхронный воркер, который запускает обработку задач параллельно.
        """
        while True:
            if not self.task_queue.empty():
                task_id, image_path, prompt, kwargs = await self.task_queue.get()
                self.task_status[task_id] = "in queue"  # Статус задачи "в очереди"
                self.logger.log(f"kwargs: {kwargs}")
                self.logger.log(f"INSIDE task_worker")
                asyncio.create_task(self._process_task(task_id, image_path, prompt, **kwargs))  # Запуск задачи параллельно
                self.logger.log(f"INSIDE task_worker TASK was CREATED")
            await asyncio.sleep(0.1)  # Задержка для предотвращения блокировки

    async def put_task(self, image_paths, prompt, **kwargs):
        """
        Добавляет задачу с несколькими изображениями в очередь.
        """
        task_id = str(uuid.uuid4())  # Уникальный ID для задачи
        self.task_status[task_id] = "in queue"  # Задача в очереди
        await self.task_queue.put((task_id, image_paths, prompt, kwargs))
        self.logger.log(f"Задача с ID {task_id} добавлена в очередь с {len(image_paths)} изображениями")
        return task_id

    async def get_task_result(self, task_id):
        """
        Получает результат выполнения задачи по её ID.
        """
        if task_id in self.result_queue:
            return {"status": "completed", "result": self.result_queue.pop(task_id)}
        elif task_id in self.task_status:
            # Проверяем статус задачи и возвращаем его
            status = self.task_status[task_id]
            if status == "in queue":
                return {"status": "in queue"}
            elif status == "in progress":
                return {"status": "in progress"}
        # Если задача не найдена
        return {"status": "not found"}