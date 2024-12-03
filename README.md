# Pytorch project template
Template to build your own pytorch projects. Focus on the implementation rather than
the boilerplate code. The toolkit is built on top of the [Hydra](https://hydra.cc/docs/intro/) framework, which allows
to have a modular configuration schema in order to build your experiment configurations easier.

It also integrates [WandB](https://docs.wandb.ai/) for experiment tracking and 
[Optuna](https://optuna.org/) for hyperparameter optimization.

Template architecture:
```
Pytorch-project-template
├── requirements.txt            # all the dependencies needed
├── src                         # source code of the project
│   ├── datasets                
│   ├── models                  
│   └── ...
├── cfgs                        # folder with the configurations for the experiments
│   ├── config.py               # check in config.py the default configuration
│   └── config.yaml
├── data                        # folder with the datasets
│   └── ...
├── evaluate.py                 # script to evaluate
└── train.py                    # script to train
             
```
## How to run experiments
Set the hyperparameters to change from the default configuration in the `config.yaml` file. Check which hyperparameters can be defined in the `config.py` file and their default values.
Use the argument `wandb` to set the name of the WandB project and log the experiment.
```
python train.py n_epochs=100 wandb=<PROJECT_NAME>
```

### Hyperparameter search
Install the Optuna plugin for Hydra.
```
pip install hydra-optuna-sweeper --upgrade
```

In order to run a hyperparameter search, use the `--multirun` flag.
```
python train.py --multirun n_epochs=100,200
```

