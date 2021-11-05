# Generalized Proximal Policy Optimization with Sample Reuse

This repository is the official implementation of the reinforcement learning algorithm Generalized Proximal Policy Optimization with Sample Reuse (GePPO), which was introduced in the [NeurIPS 2021 paper](https://arxiv.org/abs/2111.00072) with the same name.

GePPO improves the sample efficiency of the popular on-policy algorithm PPO through principled sample reuse, while still retaining PPO's approximate policy improvement guarantees. GePPO is theoretically supported by a generalized policy improvement lower bound that can be approximated using data from all recent policies.

## Requirements

The source code requires the following packages to be installed (we have included the version used to produce the results found in the paper in parentheses):

- python (3.7.7)
- gurobi (9.0.2)
- gym (0.17.1)
- matplotlib (3.1.3)
- mujoco-py (1.50.1.68)
- numpy (1.18.1)
- scipy (1.4.1)
- seaborn (0.10.1)
- tensorflow (2.1.0)

See the file `environment.yml` for the conda environment used to run our experiments, which can be built with conda using the command `conda env create`.

The MuJoCo environments used in our experiments require the MuJoCo physics engine and a MuJoCo license. Please see the [MuJoCo website](https://www.roboti.us/license.html) for more information on downloading MuJoCo and obtaining a license. 

Our implementation of GePPO uses Gurobi to determine the optimal policy weights used in the algorithm, which requires a Gurobi license. Please see the [Gurobi website](https://www.gurobi.com/downloads/) for more information on downloading Gurobi and obtaining a license. Alternatively, GePPO can be run without Gurobi by using uniform policy weights with the `--uniform` option.

## Training

Simulations can be run by calling `run` on the command line. For example, we can run simulations on the HalfCheetah-v3 environment with PPO and GePPO as follows:

```
python -m geppo.run --env_name HalfCheetah-v3 --alg_name ppo
python -m geppo.run --env_name HalfCheetah-v3 --alg_name geppo
```

By default, all algorithm hyperparameters are set to the default values used in the paper. Hyperparameters can be changed to non-default values by using the relevant option on the command line. For more information on the inputs accepted by `run`, use the `--help` option.

The results of simulations are saved in the `logs/` folder upon completion.

## Evaluation

The results of simulations saved in the `logs/` folder can be visualized by calling `plot` on the command line:

```
python -m geppo.plot --ppo_file <filename> --geppo_file <filename>
```

By default, this command saves a plot of average performance throughout training in the `figs/` folder. Other metrics can be plotted using the `--metric` option. For more information on the inputs accepted by `plot`, use the `--help` option.