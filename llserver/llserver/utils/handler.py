import requests


class UniserverHandler:
    def __init__(self, port=8000):
        self.uniserver_port = port
        ping_result = self._ping_server()
        if ping_result["status"] == "Server is working":
            print("Server is ready to go")
        else:
            print(f"Error: {ping_result}")

    def get_running_models(self):
        url = f"http://127.0.0.1:{self.uniserver_port}/running_models/"
        response = requests.get(url)
        return response.json()

    def start_model(self, model_name: str):
        url = f"http://127.0.0.1:{self.uniserver_port}/start_model/"
        params = {
            'model_name': model_name
        }
        response = requests.post(url, params=params)
        return response.json()

    def stop_model(self, model_id):
        url = f"http://127.0.0.1:{self.uniserver_port}/stop_model/"
        params = {
            'model_id': model_id
        }
        response = requests.post(url, params=params)
        return response.json()
    
    def stop_all_models(self):
        models = self.get_running_models()["running_models"]["models"]
        for model_id in models.keys():
            res = self.stop_model(model_id)
            print(res)
            
    def put_task(self, model_id, prompt, image_paths, **kwargs):
        url = f"http://127.0.0.1:{self.uniserver_port}/put_task/"

        data = {
            'model_id': model_id,
            'prompt': prompt,
            'image_paths': image_paths,
            'extra_params': kwargs
        }
        print(kwargs)
        print(data)
        response = requests.post(url, json=data)
        return response.json()
    
    def get_task_result(self, model_id, task_id):
        url = f"http://127.0.0.1:{self.uniserver_port}/get_task_result/"
        params = {
            'model_id': model_id,
            'task_id': task_id
        }
        response = requests.post(url, params=params)
        return response.json()
    
    def _ping_server(self):
        """
        Function to ping the server and check if it is working.
        """
        url = f"http://127.0.0.1:{self.uniserver_port}/"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return {"status": "Server is working"}
            else:
                return {"status": "Server is not working", "code": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"status": "Server is not working", "error": str(e)}