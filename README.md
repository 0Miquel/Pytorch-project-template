# Pytorch project template
Template to build your own pytorch projects. Focus on the implementation rather than
the boilerplate code. The toolkit is built on top of the [Hydra](https://hydra.cc/docs/intro/) framework, which allows
to have a modular configuration schema in order to build your experiment configurations easier.

It also integrates [WandB](https://docs.wandb.ai/) for experiment tracking and 
[Optuna](https://optuna.org/) for hyperparameter optimization.

The template is structured in the following way:
```
Pytorch-project-template
├── requirements.txt            # all the dependencies needed
│
├── src                         # source code of the project
│   ├── datasets                # it contains the source code of 
│   ├── models                  # all the packages of the project
│   └── ...
│
└── tools                       # scripts to run the experiments
    ├── train.py                 
    ├── evaluate.py             
    ├── cfgs                    # configuration files 
    │   ├── config.yaml 
    │   └── ...
    └── outputs                 # output of the experiments
        └── ...                
```
The source code packages contain a template class to build your own models, datasets, etc.
