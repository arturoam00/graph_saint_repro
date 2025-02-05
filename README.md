# Reproduction of the [GraphSAINT](https://arxiv.org/abs/1907.04931v4) paper 
Code for the reproduction of the GraphSAINT paper, on sampling techniques for GCNs, for the course Machine Learning for Graphs at VU Amsterdam. This code is heavily inspired in the original code of the paper, which can be found [here](https://github.com/GraphSAINT/GraphSAINT/tree/master).

## Setup
Clone the repo, cd to it and install the required dependencies (in a virtualenv) doing
```bash
$ python -m pip install -r requirements.txt
```

## Running the experiments
To run experiments you need to first download the data set(s) that you want to conduct experiments on. The authors of GraphSAINT provided a [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) for this. Unzip the files and store the data in a folder, for example `data`, under the directory of this file.

The configurations used to get the results presented on the report can be found under `conf`. You can add or modify your own `.yaml` files to tweak the hyperparameters as you like. You can see how each configuration file is supposed to look like for each of the different samplers used. `rnd` satands for random sapler, `node` for node sampler, `rw` for random walk sampler, `edge` for edge sampler and `fa` for feature aware sampler. All parameters in each of the corresponding `.yaml` configuration files are needed to be present for things to work.

To run an experiment, just a `data_prefix` path and a `config_file` are needed as arguments for the `main.py` script. For example, to run an experiment on the PPI data set (assuming this has been downloaded and extracted to a folder named `data/ppi` in this directory), do:
```bash
./main.py -d data/ppi -c conf/ppi_rnd.yaml
```
This will train a 2-layer [GraphSAGE](https://arxiv.org/abs/1706.02216v4)-like using the random node sampler.


## Outputs
When training a model, intermediate models are saved as `.pkl` files to a folder named `torch_models`, in this directory. Logs are both saved to a `.log` file (under `log_files`) and redirected to stdin. The script `mic.py` helps with extracting mic scores from the logs when training on several data sets and with several samplers.

## Example configuration file
This is an example of a configuration file that uses the random node sampler:
```yaml
# Network
dim: 512
aggr: 'concat'
loss: 'sigmoid'
arch: '1-0-1-0'
act: 'relu'
bias: 'norm'

# Training params
lr: 0.01
dropout: 0.0
n_epochs: 1000

# Sampling params
sample_coverage: 50
sampler: 'uniform_rnd'
sg_size: 6000

# Other
adj_norm: 'sym'

# Epoch
eval_train_every: 15
eval_val_every: 1

```