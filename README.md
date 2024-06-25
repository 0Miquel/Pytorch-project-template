# Pytorch project template

This toolkit integrates the most well-known frameworks for machine learning like WandB for experiment 
tracking, Hydra for configuration management or Optuna for hyperparameter search.

In addition to that, the code is meant to be fully extensible, it should be possible to
implement your own custom models/losses/optimizers without changing 
existing code.

## Overview
- [Frameworks](#frameworks)
- [Installation](#installation)
- [Train](#train)

## Frameworks
Brief introduction to the frameworks that are integrated into the toolkit.

### WandB
WandB is an experiment tracking tool for machine learning.

Using WandB in this toolkit you will be able to:
- Track the hyperparameters and metrics of every run
- Display the predictions through the different epochs
- Upload complex media and charts

For more information visit its [documentation](https://docs.wandb.ai/).


### Hydra
Hydra is a framework that simplifies the development of research and other complex applications.
It makes it possible to have a modular configuration schema in order 
to build your experiment configurations easier.

The Hydra configuration schema is the following:
```
conf
├── config.yaml             # config.yaml calls one .yaml file for 
├── dataset                 # every module inside the schema
│   ├── MNIST.yaml          # the .yaml files inside every module
│   └── cifar10.yaml        # specifies the configuration of that module   
├── model                   
│   ├── resnet.yaml         
│   └── ...   
└── ...
```
For more information visit its [documentation](https://hydra.cc/docs/intro/).


### Optuna
Optuna is an open source hyperparameter optimization framework to automate hyperparameter search

In order to use Optuna together with Hydra we first need to install the Optuna Sweeper plugin.
```commandline
pip install hydra-optuna-sweeper --upgrade
```
It is needed to set the optuna sweeper in the config file, but it is already done by default in `config.yaml`.
```yaml
defaults:
  - override hydra/sweeper: optuna
```
Further changes can be added in the config file like the number of runs to execute or
whether we want to maximize or minimize our objective metric.

## Installation
Clone the repository and install the requirements.
```commandline
git clone https://github.com/0Miquel/Pytorch-training-toolkit.git
cd Pytorch-training-toolkit
pip install -r requirements.txt
```

## Train
CLI command to run a training experiment, which runs the configuration found in `config.yaml`.
```
cd tools
python train.py 
```
Additional changes in the Hydra configuration can be added in the command line 
in the following way:
```
python train.py optimizer.settings.lr=0.1 trainer.wandb=test_project
```
In this case we have changed the default learning rate in the optimizer to 0.1 and
have set a wandb project name in order to log the results from the experiment,
if no project name is given it will not log any results into wandb.

Finally, in order to run an Optuna hyperparameter search we can use the following command:
```commandline
python train.py --multirun 'optimizer.settings.lr=choice(0.1, 0.01, 0.001, 0.0001)'
```
