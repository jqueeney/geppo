"""Creates command line parser for run.py."""
import argparse
import os
import pickle

parser = argparse.ArgumentParser()

# Setup
#########################################

setup_kwargs = [
    'runs','runs_start','cores','save_path','save_file',
    'import_path','import_file','import_idx',
    'seed','ac_seed','sim_seed',
]

parser.add_argument('--runs',help='number of trials',type=int,default=5)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)

parser.add_argument('--save_path',help='save path',type=str,default='./logs')
parser.add_argument('--save_file',help='save file name',type=str)

parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
parser.add_argument('--import_file',help='import file name',type=str)
parser.add_argument('--import_idx',help='import index',type=int,default=0)

parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--ac_seed',help='actor critic seed',type=int)
parser.add_argument('--sim_seed',help='simulation seed',type=int)

# Environment initialization
#########################################

env_kwargs = [
    'env_type','env_name','task_name','s_normalize','r_normalize',
    's_t','s_mean','s_var','r_t','r_mean','r_var'
]

parser.add_argument('--env_type',help='environment type',type=str,default='gym')
parser.add_argument('--env_name',help='environment',type=str,
    default='HalfCheetah-v3')
parser.add_argument('--task_name',help='task name (dmc only)',type=str)
parser.add_argument('--no_s_normalize',help='do not normalize observations',
    dest='s_normalize',default=True,action='store_false')
parser.add_argument('--no_r_normalize',help='do not normalize rewards',
    dest='r_normalize',default=True,action='store_false')
parser.add_argument('--s_t',help='observation filter count',type=int)
parser.add_argument('--s_mean',help='observation filter mean',type=float)
parser.add_argument('--s_var',help='observation filter var',type=float)
parser.add_argument('--r_t',help='reward filter count',type=int)
parser.add_argument('--r_mean',help='reward filter mean',type=float)
parser.add_argument('--r_var',help='reward filter var',type=float)

# Actor initialization
#########################################

actor_kwargs = [
    'actor_layers','actor_activations','actor_gain','actor_std_mult',
    'actor_per_state_std','actor_weights'
]

parser.add_argument('--actor_layers',nargs='+',
    help='list of hidden layer sizes for actor',type=int,default=[64,64])
parser.add_argument('--actor_activations',nargs='+',
    help='list of activations for actor',type=str,default=['tanh'])
parser.add_argument('--actor_gain',
    help='mult factor for final layer of actor',type=float,default=0.01)
parser.add_argument('--actor_std_mult',
    help='initial policy std deviation multiplier',type=float,default=1.0)
parser.add_argument('--actor_per_state_std',
    help='use state dependent std deviation',action='store_true')
parser.add_argument('--actor_weights',help='actor weights')

# Critic initialization
#########################################

critic_kwargs = [
    'critic_layers','critic_activations','critic_gain','critic_weights'
]

parser.add_argument('--critic_layers',nargs='+',
    help='list of hidden layer sizes for value function',
    type=int,default=[64,64])
parser.add_argument('--critic_activations',nargs='+',
    help='list of activations for value function',type=str,default=['tanh'])
parser.add_argument('--critic_gain',
    help='mult factor for final layer of value function',type=float,default=1.0)
parser.add_argument('--critic_weights',help='critic weights')

# Policy weight initialization
#########################################

polweights_kwargs = [
    'B','M_max','tradeoff','uniform'
]

parser.add_argument('--B',help='minimum batch size multiplier',
    type=int,default=2)
parser.add_argument('--M_max',help='maximum number of prior policies',
    type=int,default=20)
parser.add_argument('--tradeoff',help='sample reuse tradeoff parameter',
    type=float,default=1.0)
parser.add_argument('--uniform',help='use uniform policy weights',
    action='store_true')

# Runner initialization
#########################################

runner_kwargs = [
    'T','gamma','lam','n','vtrace','is_trunc'
]

parser.add_argument('--T',help='max steps in trajectory',
    type=int,default=1000)
parser.add_argument('--gamma',help='discount rate',
    type=float,default=0.995)
parser.add_argument('--lam',help='GAE parameter',
    type=float,default=0.97)
parser.add_argument('--n',help='minimum batch size',
    type=float,default=1024)
parser.add_argument('--no_vtrace',help='do not correct advantages w V-trace',
    dest='vtrace',default=True,action='store_false')
parser.add_argument('--is_trunc',
    help='importance sampling truncation parameter for V-trace',
    type=float,default=1.0)

# Algorithm setup
#########################################

alg_kwargs = [
    'alg_name','save_freq','checkpoint_file','keep_checkpoints'
]

parser.add_argument('--alg_name',help='algorithm',type=str,default='geppo')

parser.add_argument('--save_freq',help='how often to store temp files',
    type=float)
parser.add_argument('--checkpoint_file',help='checkpoint file name',type=str,
    default='TEMPLOG')
parser.add_argument('--keep_checkpoints',help='keep all checkpoint info',
    action='store_true')

# Training parameters
#########################################

train_kwargs = [
    'sim_size','no_op_batches'
]

parser.add_argument('--sim_size',help='length of training process',
    type=float,default=1e6)
parser.add_argument('--no_op_batches',help='number of no op batches',type=int)


# Actor and critic kwargs
#########################################

ac_kwargs = [
    'critic_lr','adv_center','adv_scale','adv_clip','update_it','nminibatch',
    'eps_ppo','eps_vary',
    'actor_lr','scaleinitlr_eps','scaleinitlr_dim','max_grad_norm','adapt_lr',
    'adapt_factor','adapt_minthresh','adapt_maxthresh','early_stop'
]

parser.add_argument('--critic_lr',
    help='value function optimizer learning rate',type=float,default=3e-4)

parser.add_argument('--no_adv_center',help='do not center advantages',
    dest='adv_center',default=True,action='store_false')
parser.add_argument('--no_adv_scale',help='do not scale advantages',
    dest='adv_scale',default=True,action='store_false')
parser.add_argument('--adv_clip',help='clipping value for advantages',
    type=float)

parser.add_argument('--update_it',help='number of epochs per update',
    type=int,default=10)
parser.add_argument('--nminibatch',
    help='number of minibatches per epoch',type=int,default=32)

parser.add_argument('--eps_ppo',help='PPO clipping parameter',
    type=float,default=0.2)
parser.add_argument('--eps_vary',
    help='vary one-step param based on penalty term calc',action='store_true')

parser.add_argument('--actor_lr',help='actor learning rate',
    type=float,default=3e-4)
parser.add_argument('--scaleinitlr_eps',
    help='scale initial LR by same factor as epsilon',action='store_true')
parser.add_argument('--scaleinitlr_dim',
    help='scale initial LR by action dimension',action='store_true')

parser.add_argument('--max_grad_norm',help='max policy gradient norm',
    type=float,default=0.5)

parser.add_argument('--no_adapt_lr',help='do not adapt LR',
    dest='adapt_lr',default=True,action='store_false')
parser.add_argument('--adapt_factor',
    help='factor used to adapt LR',type=float,default=0.03)
parser.add_argument('--adapt_minthresh',
    help='min multiple of TV for adapting LR',type=float,default=0.0)
parser.add_argument('--adapt_maxthresh',
    help='max multiple of TV for adapting LR',type=float,default=1.0)

parser.add_argument('--early_stop',help='PPO early stopping',
    action='store_true')

# For export to run.py
#########################################

def create_parser():
    return parser

all_kwargs = {
    'setup_kwargs':         setup_kwargs,
    'env_kwargs':           env_kwargs,
    'actor_kwargs':         actor_kwargs,
    'critic_kwargs':        critic_kwargs,
    'polweights_kwargs':    polweights_kwargs,
    'runner_kwargs':        runner_kwargs,
    'alg_kwargs':           alg_kwargs,
    'train_kwargs':         train_kwargs,
    'ac_kwargs':            ac_kwargs,
}

def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict

def import_inputs(inputs_dict):
    """Imports parameter info from previous simulation.
    
    Args:
        inputs_dict (dict): dictionary of inputs
    
    Returns:
        Updated input dictionary with info from previous simulation.
    """

    setup_dict = inputs_dict['setup_kwargs']

    import_path = setup_dict['import_path']
    import_file = setup_dict['import_file']
    import_idx = setup_dict['import_idx']

    import_filefull = os.path.join(import_path,import_file)
    with open(import_filefull,'rb') as f:
        import_data = pickle.load(f)

    if isinstance(import_data,list):
        assert import_idx < len(import_data), 'import_idx too large'
        import_final = import_data[import_idx]['final']
        
        import_params = import_data[import_idx]['param']
        import_params_env = import_params['env_kwargs']
        import_params_actor = import_params['actor_kwargs']
        import_params_critic = import_params['critic_kwargs']
    else:
        raise TypeError('imported data not a list')
    
    # Environment info
    env_kwargs_dict = import_params_env
    env_kwargs_dict['s_t'] = import_final['s_t']
    env_kwargs_dict['s_mean'] = import_final['s_mean']
    env_kwargs_dict['s_var'] = import_final['s_var']
    env_kwargs_dict['r_t'] = import_final['r_t']
    env_kwargs_dict['r_mean'] = import_final['r_mean']
    env_kwargs_dict['r_var'] = import_final['r_var']

    inputs_dict['env_kwargs'] = env_kwargs_dict

    # Actor info
    actor_kwargs_dict = import_params_actor
    actor_kwargs_dict['actor_weights'] = import_final['actor_weights']

    inputs_dict['actor_kwargs'] = actor_kwargs_dict

    # Critic info
    critic_kwargs_dict = import_params_critic
    critic_kwargs_dict['critic_weights'] = import_final['critic_weights']

    inputs_dict['critic_kwargs'] = critic_kwargs_dict

    return inputs_dict