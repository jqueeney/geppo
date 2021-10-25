"""Creates command line parser for run.py."""
import argparse

parser = argparse.ArgumentParser()

# Simulation and save setup

parser.add_argument('--runs',help='number of trials',type=int,default=5)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)
parser.add_argument('--save_path',help='save path',type=str,default='./logs')
parser.add_argument('--save_file',help='save file name',type=str)
parser.add_argument('--checkpoint_file',help='checkpoint file name',type=str,
    default='TEMPLOG')
parser.add_argument('--save_freq',help='how often to store temp files',
    type=float)

# Seeds

parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--ac_seed',help='actor critic seed',type=int)
parser.add_argument('--sim_seed',help='simulation seed',type=int)

# Imports

parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
parser.add_argument('--import_file',help='import file name',type=str)
parser.add_argument('--import_idx',help='import index',type=int,default=0)

# Environment initialization

parser.add_argument('--env_name',help='environment',type=str,
    default='HalfCheetah-v3')
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

# Policy weight initialization

parser.add_argument('--B',help='minimum batch size multiplier',
    type=int,default=2)
parser.add_argument('--M_max',help='maximum number of prior policies',
    type=int,default=10)
parser.add_argument('--M_targ',help='method for choosing M',
    type=str,default='ess',choices=['tv','ess','mix'])
parser.add_argument('--uniform',help='use uniform policy weights',
    action='store_true')

# Actor initialization

parser.add_argument('--actor_layers',nargs='+',
    help='list of hidden layer sizes for actor',type=int,default=[64,64])
parser.add_argument('--actor_activations',nargs='+',
    help='list of activations for actor',type=str,default=['tanh'])
parser.add_argument('--actor_gain',
    help='mult factor for final layer of actor',type=float,default=0.01)
parser.add_argument('--actor_std_mult',
    help='initial policy std deviation multiplier',type=float,default=1.0)
parser.add_argument('--actor_weights',help='actor weights')

# Critic initialization

parser.add_argument('--critic_layers',nargs='+',
    help='list of hidden layer sizes for value function',
    type=int,default=[64,64])
parser.add_argument('--critic_activations',nargs='+',
    help='list of activations for value function',type=str,default=['tanh'])
parser.add_argument('--critic_gain',
    help='mult factor for final layer of value function',type=float,default=1.0)
parser.add_argument('--critic_weights',help='critic weights')

# Runner initialization

parser.add_argument('--T',help='max steps in trajectory',
    type=int,default=1000)
parser.add_argument('--gamma',help='discount rate',
    type=float,default=0.995)
parser.add_argument('--lam',help='GAE parameter',
    type=float,default=0.97)
parser.add_argument('--n',help='minimum batch size',
    type=float,default=1024)
parser.add_argument('--is_trunc',
    help='importance sampling truncation parameter for V-trace',
    type=float,default=1.0)

# Training parameters

parser.add_argument('--alg_name',help='algorithm',
    type=str,default='geppo',choices=['ppo','geppo'])
parser.add_argument('--sim_size',help='length of training process',
    type=float,default=1e6)
parser.add_argument('--no_op_batches',help='number of no op batches',
    type=int,default=1)
parser.add_argument('--ppo_adapt',help='use adaptive LR for PPO',
    action='store_true')
parser.add_argument('--geppo_noadapt',help='do not use adaptive LR for GePPO',
    action='store_true')

# Critic kwargs

parser.add_argument('--critic_lr',
    help='value function optimizer learning rate',type=float,default=3e-4)

# Actor kwargs

parser.add_argument('--no_adv_center',help='do not center advantages',
    dest='adv_center',default=True,action='store_false')
parser.add_argument('--no_adv_scale',help='do not scale advantages',
    dest='adv_scale',default=True,action='store_false')

parser.add_argument('--actor_lr',help='actor learning rate',
    type=float,default=3e-4)
parser.add_argument('--actor_opt_type',help='actor optimizer type',
    type=str,default='Adam',choices=['Adam','SGD'])

parser.add_argument('--update_it',help='number of epochs per update',
    type=int,default=10)
parser.add_argument('--nminibatch',
    help='number of minibatches per epoch',type=int,default=32)

parser.add_argument('--eps_ppo',help='PPO clipping parameter',
    type=float,default=0.2)
parser.add_argument('--max_grad_norm',help='max policy gradient norm',
    type=float,default=0.5)

parser.add_argument('--adapt_factor',
    help='factor used to adapt LR',type=float,default=0.03)
parser.add_argument('--adapt_minthresh',
    help='min multiple of TV for adapting LR',type=float,default=0.5)
parser.add_argument('--adapt_maxthresh',
    help='max multiple of TV for adapting LR',type=float,default=1.0)

parser.add_argument('--early_stop',help='PPO TV early stopping',
    action='store_true')
parser.add_argument('--scaleinitlr',
    help='scale initial LR by same factor as epsilon',action='store_true')

def create_parser():
    return parser