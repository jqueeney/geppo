"""Helper functions and command line parser for plot.py."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse

sns.set()
sns.set_context('paper')

plot_parser = argparse.ArgumentParser()

plot_parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
plot_parser.add_argument('--ppo_file',help='file with PPO data',type=str)
plot_parser.add_argument('--geppo_file',help='file with GePPO data',type=str)
plot_parser.add_argument('--save_path',help='save path',
    type=str,default='./figs')
plot_parser.add_argument('--save_name',
    help='file name to use when saving plot',type=str,default='userplot')
plot_parser.add_argument('--metric',
    help='metric to plot',type=str,default='J_tot')
plot_parser.add_argument('--window',
    help='number of steps for plot smoothing',type=int,default=2048*100)
plot_parser.add_argument('--timesteps',help='number of steps to plot',
    type=float,default=1e6)
plot_parser.add_argument('--interval',help='how often to plot data',
    type=float,default=5e3)
plot_parser.add_argument('--se_val',
    help='standard error multiplier for plot shading',type=float,default=0.5)

def create_plotparser():
    return plot_parser

def aggregate_sim(results,x,window,metric):
    """Computes running averages for all trials."""
    sim = len(results)
    data_all = np.zeros((sim,len(x)))
    for idx in range(sim):
        log = results[idx]['train']
        samples = np.cumsum(log['steps'])
        x_filter = np.argmax(np.expand_dims(samples,1) 
            >= np.expand_dims(x,0),0)

        try:
            data_total = np.squeeze(log[metric])
        except:
            available = ', '.join(list(log.keys()))
            raise ValueError(
                '%s is not a recognized metric. Available metrics include: %s'%(
                    metric,available))

        if window > 1:
            data_totsmooth = np.convolve(np.squeeze(data_total),
                np.ones(window),'full')[:-(window-1)]        
            len_totsmooth = np.convolve(np.ones_like(data_total),
                np.ones(window),'full')[:-(window-1)]     

            data_ave = data_totsmooth / len_totsmooth   
        else:
            data_ave = data_total

        data_all[idx,:] = data_ave[x_filter]
    
    return data_all 

def open_and_aggregate(filepath,filename,x,window,metric):
    """Returns aggregated data from raw filename."""

    if filename is None:
        results = None
    else:
        with open(os.path.join(filepath,filename),'rb') as f:
            data = pickle.load(f)
        
        M = data[0]['param']['runner_kwargs']['M']
        B = data[0]['param']['runner_kwargs']['B']
        n = data[0]['param']['runner_kwargs']['n']
        if M > 1:
            b_size = n
        else:
            b_size = B * n
        
        window_batch = int(window / b_size)
        
        results = aggregate_sim(data,x,window_batch,metric)
    
    return results

def plot_compare(ppo_data,geppo_data,x,se_val,save_path,save_name):
    """Creates and saves plot."""
    
    fig, ax = plt.subplots()

    ppo_color = 'C0'
    geppo_color = 'C1'

    if ppo_data is not None:
        ppo_mean = np.mean(ppo_data,axis=0)
        if ppo_data.shape[0] > 1:
            ppo_std = np.std(ppo_data,axis=0,ddof=1)
            ppo_se = ppo_std / np.sqrt(ppo_data.shape[0])
        else:
            ppo_se = np.zeros_like(ppo_mean)

        ax.plot(x/1e6,ppo_mean,color=ppo_color,label='PPO')
        ax.fill_between(x/1e6,
            ppo_mean-se_val*ppo_se,ppo_mean+se_val*ppo_se,
            alpha=0.2,color=ppo_color)
    
    if geppo_data is not None:
        geppo_mean = np.mean(geppo_data,axis=0)
        if geppo_data.shape[0] > 1:
            geppo_std = np.std(geppo_data,axis=0,ddof=1)
            geppo_se = geppo_std / np.sqrt(geppo_data.shape[0])
        else:
            geppo_se = np.zeros_like(geppo_mean)

        ax.plot(x/1e6,geppo_mean,color=geppo_color,label='GePPO')
        ax.fill_between(x/1e6,
            geppo_mean-se_val*geppo_se,geppo_mean+se_val*geppo_se,
            alpha=0.2,color=geppo_color)

    ax.set_xlabel('Steps (M)')
    ax.legend()

    # Save plot
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    save_file = '%s_%s'%(save_name,save_date)
    os.makedirs(save_path,exist_ok=True)
    save_filefull = os.path.join(save_path,save_file)

    filename = save_filefull+'.png'
    fig.savefig(filename,bbox_inches='tight',dpi=300)