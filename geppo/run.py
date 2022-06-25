"""Entry point for RL training."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
import pickle
import copy
import multiprocessing as mp
import numpy as np
import tensorflow as tf

from geppo.common.cmd_utils import create_parser, gather_inputs, import_inputs
from geppo.envs import init_env
from geppo.algs import init_alg, gen_algs
from geppo.common.initializers import init_seeds, init_actor, init_critic
from geppo.common.optimize_weights import init_polweights
from geppo.common.runner import Runner

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run(inputs_dict):
    """Runs simulation on given seed.

    Args:
        inputs_dict (dict): dictionary of inputs
    
    Returns:
        Name of log file.
    """

    # Unpack inputs
    setup_kwargs = inputs_dict['setup_kwargs']
    env_kwargs = inputs_dict['env_kwargs']
    actor_kwargs = inputs_dict['actor_kwargs']
    critic_kwargs = inputs_dict['critic_kwargs']
    polweights_kwargs = inputs_dict['polweights_kwargs']
    runner_kwargs = inputs_dict['runner_kwargs']
    alg_kwargs = inputs_dict['alg_kwargs']
    train_kwargs = inputs_dict['train_kwargs']
    ac_kwargs = inputs_dict['ac_kwargs']

    alg_kwargs['save_path'] = setup_kwargs['save_path']
    runner_kwargs['B'] = polweights_kwargs['B']

    ac_seed = setup_kwargs['ac_seed']
    sim_seed = setup_kwargs['sim_seed']
    alg_name = alg_kwargs['alg_name']
    sim_size = train_kwargs['sim_size']
    no_op_batches = train_kwargs['no_op_batches']
    B = polweights_kwargs['B']

    # Setup
    generalize = (alg_name in gen_algs)
    polweights, M, eps_mult = init_polweights(generalize,**polweights_kwargs)
    runner_kwargs['weights'] = polweights
    runner_kwargs['M'] = M
    ac_kwargs['eps_mult'] = eps_mult

    env = init_env(**env_kwargs)

    init_seeds(ac_seed)
    actor = init_actor(env,**actor_kwargs)
    critic = init_critic(env,**critic_kwargs)
    runner = Runner(**runner_kwargs)
    alg = init_alg(sim_seed,env,actor,critic,runner,ac_kwargs,**alg_kwargs)

    # Training
    if no_op_batches is None:
        if M > 1:
            no_op_batches = B
        else:
            no_op_batches = 1
    log_name = alg.learn(sim_size,no_op_batches,inputs_dict)

    return log_name

def main():
    """Parses inputs, runs simulations, saves data."""
    start_time = datetime.now()
    
    parser = create_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(2)
    ac_seeds = np.random.SeedSequence(seeds[0]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    sim_seeds = np.random.SeedSequence(seeds[1]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]

    inputs_list = []
    for idx in range(args.runs):
        inputs_dict['alg_kwargs']['idx'] = idx
        if args.ac_seed is None:
            inputs_dict['setup_kwargs']['ac_seed'] = int(ac_seeds[idx])
        if args.sim_seed is None:
            inputs_dict['setup_kwargs']['sim_seed'] = int(sim_seeds[idx])
        
        if args.import_path and args.import_file:
            inputs_dict = import_inputs(inputs_dict)

        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        args.cores = args.runs

    with mp.get_context('spawn').Pool(args.cores) as pool:
        log_names = pool.map(run,inputs_list)
    
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
    if args.task_name is not None:
        save_env = '%s_%s'%(save_env,args.task_name.lower())
    save_alg = args.alg_name.lower()
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    if args.save_file is None:
        save_file = '%s_%s_%s_%s'%(
            args.env_type,save_env,save_alg,save_date)
    else:
        save_file = '%s_%s_%s_%s_%s'%(
            args.env_type,save_env,save_alg,args.save_file,save_date)

    os.makedirs(args.save_path,exist_ok=True)
    save_filefull = os.path.join(args.save_path,save_file)

    with open(save_filefull,'wb') as f:
        pickle.dump(outputs,f)
    
    for log_name in log_names:
        filename = os.path.join(args.save_path,log_name)
        os.remove(filename)

    end_time = datetime.now()
    print('Time Elapsed: %s'%(end_time-start_time))

if __name__=='__main__':
    main()