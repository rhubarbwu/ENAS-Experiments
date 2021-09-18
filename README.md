# ENAS Experiments

This codebase has been adapted to run experiments on improving/evaluating the robustness of the ENAS algorithm.

## Requirements

```sh
pip install -r requirements.txt
```

## Setup

Prepare the experiments in `experiments/` by running `fetch-spaces.sh`.

```sh
# default ENAS experiment set
sh fetch-spaces.sh

# poisoning ENAS experiment set
sh fetch-spaces.sh git@github.com:rusbridger/enas_poisoning.git enas_poisoning.git

# typed ENAS experiment set
sh fetch-spaces.sh git@github.com:rusbridger/enas_types.git
```

## Running

Include the set, experiment, and number of epochs as arguments. They will default to the `baseline`, `space_0`, and `300`.

```sh
# experiment from baseline set
python driver.py baseline space_0 300

# experiment from poisoning set
python driver.py enas_poisoning poisoning_0 300

# experiment from typed set
python driver.py enas_types types_0 300
```

### Checkpoints & Results

Results are CSV files stored under `results/`, also named after the set and experiment. Model checkpoints will be stored by a filename named after the set and experiment in `checkpoints/`.

- **Do not run the same experiment in two instances at once as their models will overwrite each other.** Running different experiments from the same set is okay.
- Experiments that conclude (i.e. `epoch` counter reaches `num_epoch`) will save their CSV with a timestamp in the filename at time of completion so they will not be overwritten.
- **If you're repeating an experiment that _concluded_ before, delete the previous checkpoint.**
