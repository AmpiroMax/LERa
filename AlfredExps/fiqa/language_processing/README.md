# FIQA language processing

## Ground-truth trajectories generation

In FIQA you can train [CodeT5 model](https://github.com/salesforce/codet5) using various instructions types:

- `film`. These trajectories are based on [FILM](https://github.com/soyeonm/FILM/tree/public) templates.
- `recept` (available for ALFRED only). The triplets prediction (object, receptacle, action) based on GT trajectory.
- `no_recept`. The pairs prediction (object, action) based on GT trajectory.

To generate the training sets for CodeT5 training and GT instructions for FIQA Oracle agent see [make_alfred_dataset.ipynb](make_alfred_dataset.ipynb) and [make_tfd_dataset.ipynb](make_tfd_dataset.ipynb) notebooks for ALFRED and TEACh respectively.

## Instructions processing

To obtain the processed output of the trained model for ALFRED run the following (you can change the split and the instructions type):

```
python lp_outputs.py --split valid_seen --instr_type no_recept --dataset alfred
```

For TEACh:

```
python lp_outputs.py --split valid_seen --instr_type no_recept --dataset teach
```

You can output the trained model's accuracy using the flag `--eval`:

```
python lp_outputs.py --split valid_seen --eval --instr_type no_recept --dataset alfred
```

To obtain processed GT outputs for oracle agent use the flag `--gt`:

```
python lp_outputs.py --split valid_seen --gt --instr_type no_recept --dataset alfred
```

**Note**: to use `--eval` and `--gt` flags for queried `{split}` and `{instr_type}` you'll need to generate the file with the name `{split}_gt_{instr_type}.p` using [make_alfred_dataset.ipynb](make_alfred_dataset.ipynb) and [make_tfd_dataset.ipynb](make_tfd_dataset.ipynb) notebooks for ALFRED and TEACh respectively.

