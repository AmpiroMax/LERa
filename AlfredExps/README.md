# Visual Feedback in Instruction-Following Tasks for LLM-based Embodied Agent
This is the repository containing the code from **Visual Feedback in Instruction-Following Tasks for LLM-based Embodied Agent**.

The code was tested on
```
NVIDIA TITAN RTX, Driver Version: 510.60.02
Ubuntu 20.04.6 LTS
Python 3.8.10
PyTorch 1.7.1
Torchvision 0.8.2
AI2THOR 2.1.0
```

## Installation
Clone the repository:
```
git clone [link was deleted due to double blind review]
cd eai_fiqa/
```
If you have `python3.8` (and can install potentially missing packages) on your server, you can safely ignore running a docker container. For those who haven't, we've prepared a docker image for the FIQA deployment. Build it, start a container, connect to it via ssh:
```
cd docker/
./build.sh
./start.sh
./into_via_ssh.sh
# password: user
bash
```
The container has to posess a `python3.8` version. Create a virtual environment:
```
python3.8 -m venv fiqa_venv
```
Activate it and change the `pip` version:
```
source fiqa_venv/bin/activate
pip install pip==21.3.1
```
Install the requirements:
```
pip install -r requirements.txt
```
Start the X server:
```
tmux new -s startx
sudo python3 docker/startx.py 0
(Detach from the session: Ctrl+b then d)
export DISPLAY=:0
```
Now you can check AI2THOR. The command
```
python3 docker/check_thor.py
```
must print
```
(300, 300, 3)
## Everything works!!!
```

## Additional steps to use ALFRED scenes
Download the lite version of the dataset from [here](https://github.com/askforalfred/alfred/tree/master/data) and place the `json_2.1.0` folder inside `alfred_utils/data/`.

Run the preprocessing:
```
python3 -m alfred_utils.data.obtain_preprocessed_jsons
```
The preprocessing must produce something like this:
```
Preprocessing dataset and saving to pp folders...This will take a while. Do this once as required.
Preprocessing tests_seen
100% (1533 of 1533) |####################################################################################################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
Preprocessing tests_unseen
100% (1529 of 1529) |####################################################################################################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
Preprocessing train
100% (21023 of 21023) |##################################################################################################################################################################| Elapsed Time: 0:05:01 Time:  0:05:01
Preprocessing valid_seen
100% (820 of 820) |######################################################################################################################################################################| Elapsed Time: 0:00:11 Time:  0:00:11
Preprocessing valid_unseen
100% (821 of 821) |######################################################################################################################################################################| Elapsed Time: 0:00:11 Time:  0:00:11
```

Now you can check ALFRED by running a model consisting of the random navigator and the trivial interactor:
```
python3 main.py --run_name no_recept_gt_random_nav_trivial_seg_interactor_dummy_seg_no_checker_0_0 --from_idx 0 --to_idx 0 --split valid_seen --instr_type no_recept_gt --navigator random --interactor trivial_seg_based --seg_model none --checker none --save_imgs
```
You should be able to see images of the run inside `results/valid_seen/no_recept_gt_random_nav_trivial_seg_interactor_dummy_seg_no_checker_0_0/images/0`. If you stuck at 'Resetting ThorEnv', then something went wrong, e.g. the package versions differ somehow.


## Steps to use Oracle agent
To use oracle modules, just set the corresponding options, e.g. setting `--navigator oracle` means using the oracle version of the navigator. The following command uses only ground-truth information and oracle versions of the modules:
```
python3 main.py --run_name no_replan_film_gt_instrs_oracle_nav_advanced_seg_interactor_oracle_seg_oracle_checker_0_819_changed_states_and_no_renav --from_idx 0 --to_idx 819 --split valid_seen --instr_type film --use_gt_instrs --navigator oracle --interactor advanced_seg_based --seg_model oracle --checker oracle --planner no_replan --navigator_gpu 0 --interactor_gpu 0 --debug --subdataset_type changing_states --film_use_stop_analysis
```

N.B. Due to some quirks of AI2THOR, even oracle modules can fail. For example, see [this](https://github.com/askforalfred/alfred/issues/131#issue-1536359532) issue.

## Steps to use FILM's agent
Create a folder that will contain checkpoints of the pretrained models:
```
mkdir fiqa/checkpoints/
```

### Navigator
Download the archive with the weights of FILM models from [here](https://github.com/soyeonm/FILM/tree/public#download-trained-models) (step 0), unzip it so that `Pretrained_Models_FILM` is inside `fiqa/checkpoints/`, and set `--navigator film`.

### Interactor
FILM uses two pretrained MaskRCNN models. Place the checkpoints of the models (files `alfworld_maskrcnn_objects_lr5e-3_005.pth` and `alfworld_maskrcnn_receps_lr5e-3_003.pth`) in `fiqa/checkpoints/` and set `--interactor advanced_seg_based --seg_model maskrcnn` to use FILM's interactor with MaskRCNNs.

### Subtask checker module
Set `--checker frames_diff_based` to use FILM's method of checking interaction success.

### Reproducibility
We've taken a number of measures to diminish non-determenistic results of FILM's runs.
Do not forget to set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before running an experiment:
```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
and set `--debug`.

An example of experiment with FILM:
```
python3 main.py --run_name no_replan_film_gt_instrs_film_nav_advanced_seg_interactor_maskrcnn_seg_film_checker_0_819_changed_states --from_idx 0 --to_idx 819 --split valid_seen --instr_type film --use_gt_instrs --navigator film --interactor advanced_seg_based --seg_model maskrcnn --checker frames_diff_based --planner no_replan --navigator_gpu 0 --interactor_gpu 0 --debug --film_use_stop_analysis --subdataset_type changing_states
```
