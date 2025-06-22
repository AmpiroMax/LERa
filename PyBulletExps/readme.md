# PyBullet Robotic Manipulation Project

A comprehensive robotic manipulation framework built with PyBullet for research and experimentation in robotic pick-and-place tasks using various AI models for planning and replanning.

## 🎯 Project Overview

This project implements a sophisticated robotic manipulation system that combines:
- **PyBullet physics simulation** for realistic robot-environment interaction
- **UR5e robotic arm** with Robotiq 2F85 gripper
- **AI-powered task planning** using various language models (GPT-4o, Gemini, LLaMA)
- **Computer vision** for object detection and scene understanding
- **Automated task execution** with replanning capabilities

## 🏗️ Architecture

### Core Components

- **`table_environment.py`** - Main simulation environment with UR5e robot and gripper
- **`tasks.py`** - Predefined manipulation tasks (pick-and-place, stacking, etc.)
- **`tasks_runner.py`** - Experiment runner with configuration management
- **`motion_primitives.py`** - Low-level robot control primitives
- **`utils.py`** - Utility functions for experiment management and validation
- **`custom_logger.py`** - Custom logging system for experiments

### Robot Setup

- **Robot**: UR5e 6-DOF robotic arm
- **Gripper**: Robotiq 2F85 2-finger gripper
- **Workspace**: Table environment with configurable object placement
- **Objects**: Colored blocks and bowls for manipulation tasks

## 🎮 Available Tasks

The system includes 10 predefined manipulation tasks:

1. **Pick and Place** - Place cyan block in red bowl
2. **Color Matching** - Place blocks in bowls with matching colors
3. **Easy Stacking** - Stack blocks in bowls
4. **Hard Stacking** - Complex multi-block stacking
5. **Color Rotation** - Place blocks according to color rotation pattern
6. **Advanced Color Rotation** - Two-step color rotation
7. **Two Towers** - Build two separate block towers
8. **Tower in Bowls** - Build towers inside colored bowls
9. **Pyramid of Opposites** - Build contrasting color pyramids
10. **Non-matching Bowl Pyramid** - Build pyramids in non-matching bowls

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirenments.txt
```

### Running Experiments

1. **Choose a configuration** from `exps_configs/`:
   ```bash
   python tasks_runner.py --config baseline_test_gpt4o.json
   ```

2. **Available configurations**:
   - `baseline_test_gpt4o.json` - GPT-4o baseline testing
   - `baseline_test_gemini.json` - Gemini baseline testing
   - `lera_test_*.json` - LERA (Language-guided Robotic Action) experiments
   - `ra_test_*.json` - Robotic Action experiments
   - `era_test_*.json` - ERA experiments

### Configuration Parameters

```json
{
    "experiment_name": "experiment_name",
    "replanning_model": "lera_baseline",
    "tasks_from": 1,
    "tasks_to": 10,
    "runs_per_task": 10,
    "model_name": "gpt-4o",
    "images_to_use": 1,
    "images_step": 1,
    "drop_prob": [0, 0.5],
    "do_replan": true,
    "save_video": true
}
```

## 📁 Project Structure

```
Pybullet/
├── table_environment.py      # Main simulation environment
├── tasks.py                  # Task definitions
├── tasks_runner.py           # Experiment runner
├── motion_primitives.py      # Robot control primitives
├── utils.py                  # Utility functions
├── custom_logger.py          # Logging system
├── exps_configs/             # Experiment configurations
├── ur5e/                     # UR5e robot model files
├── robotiq_2f_85/            # Gripper model files
└── bowl/                     # Bowl object models
```

## 📊 Results and Logging

- **Logs**: Stored in `logs/{experiment_name}/`
- **Results**: JSON files in `results_logs/{experiment_name}/`
- **Videos**: GIF animations of task execution
- **Metrics**: Success rates, execution times, replanning statistics
