# Project Name

This project provides a framework for managing and interacting with various AI models using a unified server (`uniserver`) and a handler for easy communication. The models are containerized and can be started, stopped, and queried for tasks and results.

## Features

- **Unified Server (`uniserver`)**: A FastAPI-based server that manages the lifecycle of AI model containers.
- **Handler**: A Python class for interacting with the `uniserver` to manage models and tasks.
- **Model Management**: Start, stop, and monitor models.
- **Task Management**: Submit tasks to models and retrieve results.

## Prerequisites

- Docker
- Python 3.8+
- FastAPI
- Uvicorn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/project-name.git
   cd project-name
   ```

2. Install the required Python packages:
   ```bash
   pip install -r dev_requirements.txt
   ```

3. Ensure Docker is installed and running on your machine.

## Usage

### Starting the Uniserver

To start the `uniserver`, run the following command:

```bash
./run_uniserver.sh
```

### Using the Handler

The `UniserverHandler` class provides an interface to interact with the `uniserver`. Below are examples of how to use it:

#### Initialize the Handler

```python
from llserver.utils.handler import UniserverHandler

# Initialize the handler with the port where the uniserver is running
handler = UniserverHandler(port=8000)
```

#### Get Running Models

```python
running_models = handler.get_running_models()
print(running_models)
```

#### Start a Model
Before starting a model, you need to build a docker image for it.

```bash
./build_model.sh --model api_model
```

Then you can start the model.
```python
response = handler.start_model("api_model")
print(response)
```

#### Stop a Model

```python
response = handler.stop_model("a2a59714-f407-4e40-a460-688793a3562f")
print(response)
```

#### Submit a Task

```python
task_id = handler.put_task(
    model_id="a2a59714-f407-4e40-a460-688793a3562f",
    image_paths=["/path/to/image1.jpg", "/path/to/image2.jpg"],
    prompt="Describe the scene in the images."
)
print(task_id)
```

#### Get Task Result

```python
result = handler.get_task_result(
    model_id="a2a59714-f407-4e40-a460-688793a3562f",
    task_id="your-task-id"
)
print(result)
```

## Configuration

- **Dockerfile**: Each model has its own Dockerfile for containerization.
- **Requirements**: Each model has a `requirements.txt` file specifying its dependencies.

