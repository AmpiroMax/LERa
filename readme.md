<div align="center">

# LERa: Replanning with Visual Feedback in Instruction Following (IROS 2025)
A Visual Language Model-based replanning approach that utilizes visual feedback for effective error correction in robotics. 🤖

> [Svyatoslav Pchelintsev*](https://github.com/)<sup>1</sup>, [Maxim Patratskiy*](https://github.com/)<sup>1</sup>, [Anatoly Onishchenko](https://github.com/)<sup>1</sup>, [Alexandr Korchemnyi](https://github.com/)<sup>1</sup>, [Aleksandr Medvedev](https://github.com/)<sup>2</sup>, [Uliana Vinogradova](https://github.com/)<sup>2</sup>, [Ilya Galuzinsky](https://github.com/)<sup>2</sup>, [Aleksey Postnikov](https://github.com/)<sup>2</sup>, [Alexey K. Kovalev](https://github.com/)<sup>1,3</sup>, [Aleksandr I. Panov](https://github.com/)<sup>1,3</sup>
> MIPT<sup>1</sup>, Sberbank of Russia, Robotics Center<sup>2</sup>, AIRI<sup>3</sup>

[\[📄Paper\]](https://arxiv.org/)  [\[🔥Project Page\]](#) [\[🚀 Quick Start\]](#-quick-start) [\[✅ Performance\]](#-performance)

[\[🔥Installation\]](#-installation) [\[🚀 Experiments\]](#-experiments) [\[🎄Custom Tasks\]](#-custom-tasks)

<!-- ![perform](.assets/teaser.png) -->

</div>

## 📋 Abstract

Large Language Models are increasingly used in robotics for task planning, but their reliance on textual inputs limits their adaptability to real-world changes and failures. To address these challenges, we propose LERa -- **L**ook, **E**xplain, **R**epl**a**n -- a Visual Language Model-based replanning approach that utilizes visual feedback. Unlike existing methods, LERa requires only a raw RGB image, a natural language instruction, an initial task plan, and failure detection—without additional information such as object detection or predefined conditions that may be unavailable in a given scenario.

The replanning process consists of three steps: (i) **Look**, where LERa generates a scene description and identifies errors; (ii) **Explain**, where it provides corrective guidance; and (iii) **Replan**, where it modifies the plan accordingly. LERa is adaptable to various agent architectures and can handle errors from both dynamic scene changes and task execution failures.

We evaluate LERa on the newly introduced **ALFRED-ChaOS** and **VirtualHome-ChaOS** datasets, achieving a **40% improvement** over baselines in dynamic environments. In tabletop manipulation tasks with a predefined probability of task failure within the PyBullet simulator, LERa improves success rates by up to **67%**.

## 🎯 Key Features

- **🔍 Visual Feedback**: Uses only raw RGB images without requiring object detection or masks
- **🔄 Three-Step Process**: Look → Explain → Replan for comprehensive error understanding
- **🏗️ Architecture Agnostic**: Works with any agent that provides failure detection
- **🌍 Multi-Environment**: Tested on ALFRED, VirtualHome, PyBullet, and real robots
- **📈 Significant Improvements**: Up to 67% success rate improvement over baselines
- **🤖 Real-World Ready**: Validated on physical tabletop manipulation robot

## 🏆 Performance

| Environment | Baseline | LERa | Improvement |
|-------------|----------|------|-------------|
| ALFRED-ChaOS (Seen) | 33.04% | **49.55%** | +16.51% |
| ALFRED-ChaOS (Unseen) | 31.65% | **53.60%** | +21.95% |
| VirtualHome-ChaOS | 50.00% | **94.06%** | +44.06% |
| PyBullet (GPT-4o) | 19.00% | **67.00%** | +48.00% |
| PyBullet (Gemini) | 19.00% | **86.00%** | +67.00% |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker
- NVIDIA GPU (for AI2THOR)
- PyTorch 1.7.1+

### Installation

**Each module must be installed separately according to its own README:**

1. **AlfredExps** - see `AlfredExps/README.md`
2. **PyBulletExps** - see `PyBulletExps/readme.md` 
3. **llserver** - see `llserver/README.md`

### Running Experiments

#### 1. Start llserver (required first!)

```bash
cd llserver
./run_uniserver.sh
```

#### 2. Build and start LERa model

```bash
cd llserver
./build.sh --model lera_api

# In new terminal
python -c "
from llserver.utils.handler import UniserverHandler
handler = UniserverHandler(port=8000)
response = handler.start_model('lera_api')
print('Model started:', response)
"
```

#### 3. Run experiments

**AlfredExps** - examples from `runner.txt`:
```bash
cd AlfredExps

# Oracle agent with replanning
python3 main.py --run_name val_seen_replan_oracle_eccv_0_819 \
    --from_idx 0 --to_idx 819 --split valid_seen \
    --instr_type film --use_gt_instrs \
    --navigator oracle --interactor advanced_seg_based \
    --seg_model oracle --checker oracle \
    --planner with_replan \
    --navigator_gpu 0 --interactor_gpu 0 \
    --debug --subdataset_type changing_states --save_imgs
```

**PyBulletExps** - configurations in `exps_configs/`:
```bash
cd PyBulletExps

# Example configuration lera_test_gemini.json:
{
    "experiment_name": "test_gemini_pro_all",
    "tasks_from": 1,
    "tasks_to": 10,
    "runs_per_task": 10,
    "model_name": "gemini-pro-1.5",
    "images_to_use": 1,
    "images_step": 1,
    "drop_prob": [0, 0.5],
    "do_replan": true,
    "save_video": true
}

# Run experiment
python tasks_runner.py --config lera_test_gemini.json
```

## 📁 Project Structure

```
Lera-Replanning/
├── AlfredExps/          # AI2THOR virtual environment experiments
│   └── fiqa/            # Main agent logic (planner, navigator, interactor)
├── PyBulletExps/        # Physical robot simulation experiments
│   ├── table_environment.py  # Main simulation environment
│   ├── tasks.py              # Task definitions
│   ├── tasks_runner.py       # Experiment runner
│   └── exps_configs/         # Experiment configurations
└── llserver/            # LLM model management server
    ├── server/          # FastAPI server with Docker containers
    ├── models/          # LERA API, LERA Baseline, various LLMs
    └── utils/           # Handler and utilities
```

## 🎮 Available Environments

### 🏠 ALFRED-ChaOS
- **Purpose**: Dynamic object state changes in household tasks
- **Tasks**: "Put a clean sponge on a metal rack", etc.
- **Challenge**: Objects change states during execution
- **Results**: 40%+ improvement over baselines

### 🏡 VirtualHome-ChaOS  
- **Purpose**: Container state changes (microwave, refrigerator, dishwasher)
- **Tasks**: Heating, cooling, washing objects
- **Challenge**: Container doors open/close unexpectedly
- **Results**: 94% success rate

### 🤖 TableTop-PyBullet
- **Purpose**: Action execution failures in manipulation
- **Tasks**: 10 different pick-and-place and stacking tasks
- **Challenge**: Objects drop with predefined probability
- **Results**: Up to 67% improvement

### 🦾 TableTop-RoboticStand
- **Purpose**: Real-world validation
- **Platform**: XArm6 with RealSense sensors
- **Tasks**: 18 different manipulation tasks
- **Results**: 15/18 successful replanning trials

## 📊 Configuration Options

### AlfredExps
- **Planners**: `no_replan`, `with_replan`
- **Navigators**: `oracle`, `film`, `random`
- **Interactors**: `oracle`, `advanced_seg_based`, `trivial_seg_based`
- **Segmentation**: `oracle`, `maskrcnn`, `segformer`, `none`

### PyBulletExps
- **Models**: `gpt-4o`, `gemini`, `llama-11b`, `llama-90b`
- **Strategies**: `lera_baseline`, `lera_api`, `ra_baseline`, `era_baseline`
- **Tasks**: 10 different manipulation tasks

## 📈 Results

Results are saved in:
- **AlfredExps**: `results/{split}/{run_name}/`
- **PyBulletExps**: `results_logs/{experiment_name}/`
- **llserver**: Logs in Docker containers

## 📝 Notes

- For reproducibility, use `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Oracle modules may occasionally fail due to AI2THOR quirks
- All experiments support logging and image saving
- PyBullet experiments include GIF animations of task execution

## 📄 Citation

```bibtex
@article{pchelintsev2025lera,
  title={LERa: Replanning with Visual Feedback in Instruction Following},
  author={Pchelintsev, Svyatoslav and Patratskiy, Maxim and Onishchenko, Anatoly and Korchemnyi, Alexandr and Medvedev, Aleksandr and Vinogradova, Uliana and Galuzinsky, Ilya and Postnikov, Aleksey and Kovalev, Alexey K. and Panov, Aleksandr I.},
  journal={IROS},
  year={2025}
}
```