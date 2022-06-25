"""Helper functions used by Runner class."""
import numpy as np
import scipy.signal as sp_sig

def aggregate_data(data):
    return np.concatenate([np.concatenate(datum,0) for datum in data],0)

def discounted_sum(x,rate):
    return sp_sig.lfilter([1], [1, float(-rate)], x[::-1], axis=0)[::-1]

def gae_vtrace(s,a,sp,r,d,neglogp,gamma,lam,is_trunc,actor,critic):
    """Calculates off-policy GAE with V-trace for trajectory."""

    neglogp_pik = actor.neglogp_pik(s,a)

    ratio = np.exp(neglogp - neglogp_pik)
    ratio_trunc = np.minimum(ratio,is_trunc)

    n = ratio.shape[0]
    ones_U = np.triu(np.ones((n,n)),0)
    
    rate_L = np.tril(np.ones((n,n))*gamma*lam,-1)
    rates = np.tril(np.cumprod(rate_L+ones_U,axis=0),0)

    ratio_trunc_repeat = np.repeat(np.expand_dims(ratio_trunc,1),n,axis=1)
    ratio_trunc_L = np.tril(ratio_trunc_repeat,-1)
    ratio_trunc_prods = np.tril(np.cumprod(ratio_trunc_L+ones_U,axis=0),0)

    V = critic.value(s).numpy()
    Vp = critic.value(sp).numpy()

    delta = r + gamma * (1-d) * Vp - V

    intermediate = rates * ratio_trunc_prods * np.expand_dims(delta,axis=1)
    adv = np.sum(intermediate,axis=0)
    rtg = adv * ratio_trunc + V

    adv = adv.astype('float32')
    rtg = rtg.astype('float32')

    return adv, rtg

def gae(s,sp,r,d,gamma,lam,critic):
    """Calculates on-policy GAE for trajectory."""

    V = critic.value(s).numpy()
    Vp = critic.value(sp).numpy()

    delta = r + gamma * (1-d) * Vp - V

    adv = discounted_sum(delta,gamma*lam)
    rtg = adv + V

    adv = adv.astype('float32')
    rtg = rtg.astype('float32')

    return adv, rtg

def gae_batch(s_batch,a_batch,sp_batch,r_batch,d_batch,neglogp_batch,
    gamma,lam,vtrace,is_trunc,actor,critic):
    """Calculates advantage estimates for a single batch."""
    
    traj_count = len(r_batch)

    adv_batch = []
    rtg_batch = []
    for idx in range(traj_count):
        s_traj = s_batch[idx]
        a_traj = a_batch[idx]
        sp_traj = sp_batch[idx]
        r_traj = r_batch[idx]
        d_traj = d_batch[idx]
        neglogp_traj = neglogp_batch[idx]

        if vtrace:
            adv_traj, rtg_traj = gae_vtrace(s_traj,a_traj,sp_traj,r_traj,d_traj,
                neglogp_traj,gamma,lam,is_trunc,actor,critic)
        else:
            adv_traj, rtg_traj = gae(s_traj,sp_traj,r_traj,d_traj,
                gamma,lam,critic)
        adv_batch.append(adv_traj)
        rtg_batch.append(rtg_traj)
    
    return adv_batch, rtg_batch

def gae_all(s_all,a_all,sp_all,r_all,d_all,neglogp_all,gamma,lam,vtrace,
    is_trunc,actor,critic):
    """Calculates advantage estimates for all batches."""
    
    batch_count = len(r_all)

    adv_all = []
    rtg_all = []
    for idx in range(batch_count):
        s_batch = s_all[idx]
        a_batch = a_all[idx]
        sp_batch = sp_all[idx]
        r_batch = r_all[idx]
        d_batch = d_all[idx]
        neglogp_batch = neglogp_all[idx]

        adv_batch, rtg_batch = gae_batch(s_batch,a_batch,sp_batch,r_batch,
            d_batch,neglogp_batch,gamma,lam,vtrace,is_trunc,actor,critic)
        adv_all.append(adv_batch)
        rtg_all.append(rtg_batch)
    
    return adv_all, rtg_all

def reward_calc(r_raw,gamma):
    """Calculates objective value and reward-to-go."""
    rtg_raw = discounted_sum(r_raw,gamma)

    J_tot = np.sum(r_raw)
    J_disc = rtg_raw[0]

    rtg_raw = rtg_raw.astype('float32')
    J_tot = J_tot.astype('float32')
    J_disc = J_disc.astype('float32')

    return rtg_raw, J_tot, J_disc