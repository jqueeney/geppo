# Generalized Proximal Policy Optimization with Sample Reuse

**Any future code updates will appear in the repository <https://github.com/jqueeney/gpi>.**

This repository is the official implementation of the reinforcement learning algorithm Generalized Proximal Policy Optimization with Sample Reuse (GePPO), which was introduced in the [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper/2021/hash/63c4b1baf3b4460fa9936b1a20919bec-Abstract.html) with the same name. Any future code updates will appear in the repository <https://github.com/jqueeney/gpi>, which contains implementations of several [Generalized Policy Improvement algorithms](https://arxiv.org/abs/2206.13714) including GePPO.

GePPO improves the sample efficiency of the popular on-policy algorithm PPO through principled sample reuse, while still retaining PPO's approximate policy improvement guarantees. GePPO is theoretically supported by a generalized policy improvement lower bound that can be approximated using data from all recent policies.

Please consider citing our paper as follows:

```
@inproceedings{queeney_2021_geppo,
 author = {Queeney, James and Paschalidis, Ioannis Ch. and Cassandras, Christos G.},
 title = {Generalized Proximal Policy Optimization with Sample Reuse},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 volume = {34},
 year = {2021}
}
```

## Requirements

The source code requires the following packages to be installed (we have included the latest version used to test the code in parentheses):

- python (3.8.13)
- dm-control (1.0.0)
- gurobi (9.5.1)
- gym (0.21.0)
- matplotlib (3.5.1)
- mujoco-py (1.50.1.68)
- numpy (1.22.3)
- scipy (1.8.0)
- seaborn (0.11.2)
- tensorflow (2.7.0)

See the file `environment.yml` for the latest conda environment used to run our code, which can be built with conda using the command `conda env create`.

Some OpenAI Gym environments and all DeepMind Control Suite environments require the MuJoCo physics engine. Please see the [MuJoCo website](https://mujoco.org/) for more information. 

Our implementation of GePPO uses Gurobi to determine the optimal policy weights used in the algorithm, which requires a Gurobi license. Please see the [Gurobi website](https://www.gurobi.com/downloads/) for more information on downloading Gurobi and obtaining a license. Alternatively, GePPO can be run without Gurobi by using uniform policy weights with the `--uniform` option.

## Training

Simulations can be run by calling `run` on the command line. See below for examples of running PPO and GePPO on both OpenAI Gym and DeepMind Control Suite environments:

```
python -m geppo.run --env_type gym --env_name HalfCheetah-v3 --alg_name ppo
python -m geppo.run --env_type gym --env_name HalfCheetah-v3 --alg_name geppo

python -m geppo.run --env_type dmc --env_name cheetah --task_name run --alg_name ppo
python -m geppo.run --env_type dmc --env_name cheetah --task_name run --alg_name geppo
```

Hyperparameters can be changed to non-default values by using the relevant option on the command line. For more information on the inputs accepted by `run`, use the `--help` option.

The results of simulations are saved in the `logs/` folder upon completion.

## Evaluation

The results of simulations saved in the `logs/` folder can be visualized by calling `plot` on the command line:

```
python -m geppo.plot --ppo_file <filename> --geppo_file <filename>
```

By default, this command saves a plot of average performance throughout training in the `figs/` folder. Other metrics can be plotted using the `--metric` option. For more information on the inputs accepted by `plot`, use the `--help` option.
