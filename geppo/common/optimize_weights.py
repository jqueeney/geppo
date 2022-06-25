import numpy as np

def init_polweights(generalize,B,M_max,tradeoff=1.0,uniform=False,
    truncate=0.01):
    """Calculates M, policy weights, and generalized parameter multiple.
    
    Args:
        generalize (bool): if True, consider sample reuse
        B (int): multiple of minimum batch size in on-policy case
        M_max (int): maximum number of prior policies
        tradeoff (float): relative importance of update size vs. ESS in [0,1]
        uniform (bool): if True, use uniform weights
    
    Returns:
        weights (np.ndarray): optimized policy weights
        M (int): number of prior policies
        eps_mult (float): multiple of on-policy penalty parameter to use
    """
    assert tradeoff >= 0.0, 'tradeoff must be in [0,1]'
    assert tradeoff <= 1.0, 'tradeoff must be in [0,1]'

    if generalize:
        if uniform:
            weights, M, eps_mult = polweights_uniform(B,tradeoff)
        else:
            weights, M, eps_mult = polweights_optimal(B,M_max,tradeoff,truncate)
    else:
        weights = np.ones(1)
        M = 1
        eps_mult = 1.0        
    
    return weights, M, eps_mult

def polweights_uniform(B,tradeoff):
    """Finds optimal uniform policy weights."""

    M_min = B
    M_max = 2 * B - 1
    
    M_all = np.arange(M_max-M_min+1) + M_min

    ess_val = (1/M_all - 1/M_max) / (1/M_min - 1/M_max)
    tv_val = (M_all - M_min) / (M_max - M_min)

    M = np.argmin(tradeoff*ess_val+(1-tradeoff)*tv_val) + M_min

    weights = np.ones(M) / M

    eps_mult = 2 / (M+1)

    return weights, M, eps_mult

def polweights_optimal(B,M_max,tradeoff,truncate=0.01):
    """Finds optimal policy weights."""
    import gurobipy as gp

    ones = np.ones(M_max)
    increase = np.arange(M_max) + 1

    # Normalize objective
    if tradeoff > 0.0 and tradeoff < 1.0:
        # Recursively call without objective normalization or truncation
        weights_tv, _, _ = polweights_optimal(B,M_max,0.0,0.0)
        weights_ess, _, _ = polweights_optimal(B,M_max,1.0,0.0)

        tv_min = np.dot(increase,weights_tv)
        tv_max = B

        ess_min = np.sum(np.square(weights_ess))
        ess_max = 1/B

        objweight_ess = tradeoff / (ess_max - ess_min)
        objweight_tv = (1-tradeoff) / (tv_max - tv_min)
    else:
        objweight_ess = tradeoff
        objweight_tv = (1-tradeoff)

    # Optimize weights
    obj_tv = objweight_tv * increase
    obj_ess = np.identity(M_max) * objweight_ess

    model = gp.Model()
    model.Params.OutputFlag = 0

    p_opt = model.addMVar((M_max,),name='weights')

    model.setObjective(p_opt @ obj_ess @ p_opt + obj_tv @ p_opt)

    model.addConstr(ones @ p_opt == 1)
    model.addConstr(increase @ p_opt <= B)
    model.addConstr(p_opt @ p_opt <= (1/B))

    model.optimize()
    if model.Status != 2:
        raise ValueError(
            'Gurobi unable to find optimal weights (status %d)'%model.Status)

    weights = p_opt.X

    # Truncate weights and calculate relevant quantities
    active = weights > truncate
    weights = weights[active]
    weights = weights / np.sum(weights)

    eps_mult = 1 / np.dot(weights,increase[active])
        
    M = len(weights)

    return weights, M, eps_mult

