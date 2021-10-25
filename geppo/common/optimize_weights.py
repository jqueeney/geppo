import numpy as np

def init_polweights(alg_name,B,M_max,M_targ='ess',uniform=False):
    """Calculates M, policy weights, and clipping param multiple for GePPO.
    
    Args:
        alg_name (str): ppo or geppo
        B (int): multiple of minimum batch size in PPO
        M_max (int): maximum number of prior policies
        M_targ (str): what to optimize weights for (tv, ess or mix)
        uniform (bool): if True, use uniform weights
    
    Returns:
        weights (np.ndarray): optimized policy weights
        M (int): number of prior policies
        eps_mult (float): multiple of PPO clipping parameter to use
    """
    if alg_name == 'ppo':
        weights = np.ones(1)
        M = 1
        eps_mult = 1.0
    else:
        if uniform:
            if M_targ == 'tv':
                M = B
            elif M_targ == 'ess':
                M = 2*B - 1
            elif M_targ == 'mix':
                M = int(np.ceil(np.mean([B,2*B-1])))
            else:
                raise ValueError('M_targ must be tv, ess or mix')
            
            weights = np.ones(M) / M
            eps_mult = 2 / (M+1)
        else:
            if M_targ == 'tv':
                weights = optweights_tv(B,M_max)
            elif M_targ == 'ess':
                weights = optweights_ess(B,M_max)
            elif M_targ == 'mix':
                weights_tv = optweights_tv(B,M_max)
                weights_ess = optweights_ess(B,M_max)
                weights = 0.5 * weights_tv + 0.5 * weights_ess
            else:
                raise ValueError('M_targ must be tv, ess or mix')

            active = weights > 0.01
            weights = weights[active]
            weights = weights / np.sum(weights)

            increase = np.arange(M_max) + 1
            eps_mult = 1 / np.dot(weights,increase[active])
            M = len(weights)
    
    return weights, M, eps_mult

def optweights_tv(B,M_max):
    """Finds optimal weights to maximize total variation distance."""
    import gurobipy as gp

    ones = np.ones(M_max)
    increase = np.arange(M_max) + 1

    m_tv = gp.Model()
    m_tv.Params.OutputFlag = 0

    p_tv = m_tv.addMVar((M_max,),name='weights')

    m_tv.setObjective(increase @ p_tv)

    m_tv.addConstr(ones @ p_tv == 1)
    m_tv.addConstr(p_tv @ p_tv <= (1/B))

    m_tv.optimize()
    if m_tv.Status != 2:
        raise ValueError(
            'Gurobi unable to find optimal weights (status %d)'%m_tv.Status)
    
    weights = p_tv.X

    return weights

def optweights_ess(B,M_max):
    """Finds optimal weights to maximize effective sample size (ESS)."""
    import gurobipy as gp
    
    ones = np.ones(M_max)
    increase = np.arange(M_max) + 1

    m_ess = gp.Model()
    m_ess.Params.OutputFlag = 0

    p_ess = m_ess.addMVar((M_max,),name='weights')

    m_ess.setObjective(p_ess @ p_ess)

    m_ess.addConstr(ones @ p_ess == 1)
    m_ess.addConstr(increase @ p_ess == B)

    m_ess.optimize()
    if m_ess.Status != 2:
        raise ValueError(
            'Gurobi unable to find optimal weights (status %d)'%m_ess.Status)

    weights = p_ess.X

    return weights