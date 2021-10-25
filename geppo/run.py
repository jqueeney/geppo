"""Entry point for RL training."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import tensorflow as tf

from geppo.common.cmd_utils import create_parser
from geppo.common.initializers import init_seeds, init_env
from geppo.common.initializers import init_actor, init_critic
from geppo.common.initializers import import_params
from geppo.common.optimize_weights import init_polweights
from geppo.common.runner import Runner

from geppo.algs import GePPO

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    input_keys = ['save_path','checkpoint_file','save_freq',
        'ac_seed','sim_seed','env_name','s_normalize','r_normalize',
        's_t','s_mean','s_var','r_t','r_mean','r_var',
        'B','M_max','M_targ','uniform',
        'actor_layers','actor_activations',
        'actor_gain','actor_std_mult','actor_weights',
        'critic_layers','critic_activations','critic_gain','critic_weights',
        'T','gamma','lam','n','is_trunc',
        'alg_name','sim_size','no_op_batches','ppo_adapt','geppo_noadapt'
    ]

    args_dict = vars(args)
    inputs_dict = dict()
    for key in input_keys:
        inputs_dict[key] = args_dict[key]

    if args.import_path and args.import_file:
        setup_dict = import_params(
            args.import_path,args.import_file,args.import_idx)
        for key in setup_dict.keys():
            inputs_dict[key] = setup_dict[key]

    ac_keys = ['critic_lr','adv_center','adv_scale',
        'actor_lr','actor_opt_type','update_it','nminibatch',
        'eps_ppo','max_grad_norm',
        'adapt_factor','adapt_minthresh','adapt_maxthresh',
        'early_stop','scaleinitlr']
    ac_kwargs = dict()
    for key in ac_keys:
        ac_kwargs[key] = args_dict[key]
    inputs_dict['ac_kwargs'] = ac_kwargs

    return inputs_dict

def run(idx,save_path,checkpoint_file,save_freq,ac_seed,sim_seed,
    alg_name,env_name,s_normalize,r_normalize,
    s_t,s_mean,s_var,r_t,r_mean,r_var,
    B,M_max,M_targ,uniform,
    actor_layers,actor_activations,actor_gain,actor_std_mult,actor_weights,
    critic_layers,critic_activations,critic_gain,critic_weights,
    T,gamma,lam,n,is_trunc,
    sim_size,no_op_batches,ppo_adapt,geppo_noadapt,
    ac_kwargs
    ):
    """Runs simulation on given seed. 
    
    See command line parser for information on inputs.
    """

    # Save input parameters as dict
    params = locals()

    # Setup
    env = init_env(env_name,s_normalize,r_normalize,
        s_t,s_mean,s_var,r_t,r_mean,r_var)
    
    init_seeds(ac_seed)
    actor = init_actor(env,actor_layers,actor_activations,actor_gain,
        actor_std_mult,actor_weights)

    critic = init_critic(env,critic_layers,critic_activations,critic_gain,
        critic_weights)
    
    polweights, M, eps_mult = init_polweights(alg_name,B,M_max,M_targ,uniform)
    params['polweights'] = polweights
    params['M'] = M
    ac_kwargs['eps_mult'] = eps_mult

    if alg_name == 'ppo':
        b_size = B * n
        if ppo_adapt:
            ac_kwargs['adaptlr'] = True
        else:
            ac_kwargs['adaptlr'] = False
    else:
        b_size = n
        if geppo_noadapt:
            ac_kwargs['adaptlr'] = False
        else:
            ac_kwargs['adaptlr'] = True
    params['b_size'] = b_size

    runner = Runner(T,gamma,lam,b_size,is_trunc,M,polweights)

    alg = GePPO(sim_seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file)
    
    # Training
    log_name = alg.learn(sim_size,no_op_batches,params)

    return log_name

def run_wrapper(inputs_dict):
    return run(**inputs_dict)

def main():
    """Parses inputs, runs simulations, saves data."""
    parser = create_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(2)
    ac_seeds = np.random.SeedSequence(seeds[0]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    sim_seeds = np.random.SeedSequence(seeds[1]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]

    inputs_list = []
    for run in range(args.runs):
        inputs_dict['idx'] = run
        if args.ac_seed is None:
            inputs_dict['ac_seed'] = int(ac_seeds[run])
        if args.sim_seed is None:
            inputs_dict['sim_seed'] = int(sim_seeds[run])

        inputs_list.append({**inputs_dict})

    if args.cores is None:
        args.cores = args.runs

    with mp.get_context('spawn').Pool(args.cores) as pool:
        log_names = pool.map(run_wrapper,inputs_list)
    
    # Aggregate results
    outputs = []
    for log_name in log_names:
        os.makedirs(args.save_path,exist_ok=True)
        filename = os.path.join(args.save_path,log_name)
        
        with open(filename,'rb') as f:
            log_data = pickle.load(f)
        
        outputs.append(log_data)

    # Save data
    save_env = args.env_name.split('-')[0].lower()
    save_alg = args.alg_name.lower()
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    if args.save_file is None:
        save_file = '%s_%s_%s'%(save_env,save_alg,save_date)
    else:
        save_file = '%s_%s_%s_%s'%(save_env,save_alg,args.save_file,save_date)

    os.makedirs(args.save_path,exist_ok=True)
    save_filefull = os.path.join(args.save_path,save_file)

    with open(save_filefull,'wb') as f:
        pickle.dump(outputs,f)
    
    for log_name in log_names:
        filename = os.path.join(args.save_path,log_name)
        os.remove(filename)


if __name__=='__main__':
    main()