# ENAS Experiments

This codebase has been adapted to run experiments on improving/evaluating the robustness of the ENAS algorithm. Thanks to [Melody Guan](https://www.linkedin.com/in/melodyguan) for the [basis of work](https://github.com/melodyguan/enas).

## Requirements

```sh
pip install -r requirements.txt
```

## Setup

Prepare the experiments in `experiments/`. You can name `<experiment-set-name>` whatever you like. We provide the following experiment sets for `<experiment-set-url>`.

- Search Space Poisoning (SSP): [`poisoning`](experiments/poisoning)
  - [Poisoning the Search Space in Neural Architecture Search](https://openreview.net/forum?id=fB3z4GrHCYv) [[arXiv](https://arxiv.org/abs/2106.14406)]
  - [Towards One Shot Search Space Poisoning in Neural Architecture Search](https://ojs.aaai.org/index.php/AAAI/article/view/21658) [[arXiv](https://arxiv.org/abs/2111.07138)]
- Typed NAS: [`neuralarts`](experiments/neuralarts)
  - [NeuralArTS: Structuring Neural Architecture Search with Type Theory](https://ojs.aaai.org/index.php/AAAI/article/view/21679) [[arXiv](https://arxiv.org/abs/2110.08710)]

## Running

Include the set, experiment, and number of epochs as arguments. They will default to the `baseline`, `space_0`, and `300`.

```sh
# experiment from baseline set
python driver.py baseline space_0 300

# experiment from poisoning set
python driver.py poisoning poisoning_0 300

# experiment from typed set
python driver.py neuralarts types_0 300
```

### Checkpoints & Results

Results are CSV files stored under `results/`, also named after the set and experiment. Model checkpoints will be stored by a filename named after the set and experiment in `checkpoints/`.

- **Do not run the same experiment in two instances at once as their models will overwrite each other.** Running different experiments from the same set is okay.
- Experiments that conclude (i.e. `epoch` counter reaches `num_epoch`) will save their CSV with a timestamp in the filename at time of completion so they will not be overwritten.
- **If you're repeating an experiment that _concluded_ before, delete the previous checkpoint.**
