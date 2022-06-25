import numpy as np

from geppo.common.runner_utils import aggregate_data, gae_all, reward_calc

class Runner:
    """Class for running simulations and storing recent simulation data."""

    def __init__(self,T,gamma,lam,B,n,vtrace,is_trunc,M,weights):
        """Initializes Runner class.

        Args:
            T (int): maximum episode length
            gamma (float): discount rate
            lam (float): Generalized Advantage Estimation parameter lambda
            B (int): multiple of minimum batch size
            n (float): minimum batch size
            vtrace (bool): if True, correct advantages w V-trace
            is_trunc (float): importance sampling truncation parameter for 
                off-policy advantage estimation
            M (int): number of prior policies
            weights (np.ndarray): policy weights
        """

        self.T = T
        self.gamma = gamma
        self.lam = lam
        
        if M > 1:
            self.b_size = n
        else:
            self.b_size = B * n

        self.vtrace = vtrace
        self.is_trunc = is_trunc

        self.noldpols = M - 1
        self.weights = weights

        self.reset()
    
    def reset(self):
        self.reset_buffer()
        self.reset_cur()
    
    def update(self):
        if self.noldpols > 0:
            self.update_buffer()
        self.reset_cur()
    
    def reset_cur(self):
        """Resets data storage for current batch."""
        self.s_batch = []
        self.s_raw_batch = []
        self.a_batch = []
        self.r_batch = []
        self.sp_batch = []
        self.d_batch = []
        self.neglogp_batch = []
        self.klinfo_batch = []
        self.k_batch = []
        self.rtg_raw_batch = []
        self.J_tot_batch = []
        self.J_disc_batch = []
        self.traj_len_batch = []

        self.traj_total = 0
        self.steps_total = 0
    
    def reset_buffer(self):
        """Resets data storage for buffer."""
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []    
        self.sp_buffer = []
        self.d_buffer = []
        self.neglogp_buffer = []
        self.klinfo_buffer = []
        self.k_buffer = []

        self.k = 0
        self.next_idx = 0
        self.full_buffer = False
    
    def update_buffer(self):
        """Updates buffer with data from current batch."""
        if self.next_idx >= len(self.s_buffer):
            self.s_buffer.append(self.s_batch)
            self.a_buffer.append(self.a_batch)
            self.r_buffer.append(self.r_batch)
            self.sp_buffer.append(self.sp_batch)
            self.d_buffer.append(self.d_batch)
            self.neglogp_buffer.append(self.neglogp_batch)
            self.klinfo_buffer.append(self.klinfo_batch)
            self.k_buffer.append(self.k_batch)
        else:
            self.s_buffer[self.next_idx] = self.s_batch
            self.a_buffer[self.next_idx] = self.a_batch
            self.r_buffer[self.next_idx] = self.r_batch
            self.sp_buffer[self.next_idx] = self.sp_batch
            self.d_buffer[self.next_idx] = self.d_batch
            self.neglogp_buffer[self.next_idx] = self.neglogp_batch
            self.klinfo_buffer[self.next_idx] = self.klinfo_batch
            self.k_buffer[self.next_idx] = self.k_batch

        self.k += 1
        self.next_idx = (self.next_idx + 1) % self.noldpols

        if len(self.s_buffer) == self.noldpols:
            self.full_buffer = True
        else:
            self.full_buffer = False

    def _generate_traj(self,env,actor):
        """Generates single trajectory."""
        s_traj = []
        s_raw_traj = []
        a_traj = []
        r_traj = []
        r_raw_traj = []
        sp_traj = []
        d_traj = []
        neglogp_traj = []

        full = True

        s = env.reset()
        s_raw,_ = env.get_raw()

        for t in range(self.T):
            s_old = s
            s_old_raw = s_raw

            a = actor.sample(s_old).numpy()
            neglogp = actor.neglogp(s_old,a).numpy()

            s, r, d, _ = env.step(actor.clip(a))
            s_raw, r_raw = env.get_raw()

            if t == (self.T-1):
                d = False

            # Store
            s_traj.append(s_old)
            s_raw_traj.append(s_old_raw)
            a_traj.append(a)
            r_traj.append(r)
            r_raw_traj.append(r_raw)
            sp_traj.append(s)
            d_traj.append(d)
            neglogp_traj.append(neglogp)

            self.steps_batch += 1

            if d:
                break

            if self.steps_batch >= self.b_size:
                if t < (self.T-1):
                    full = False
                break

        s_traj = np.array(s_traj)
        s_raw_traj = np.array(s_raw_traj)
        a_traj = np.array(a_traj)
        r_traj = np.array(r_traj)
        r_raw_traj = np.array(r_raw_traj)
        sp_traj = np.array(sp_traj)
        d_traj = np.array(d_traj)
        neglogp_traj = np.array(neglogp_traj)
        klinfo_traj = actor.get_kl_info(s_traj)
        k_traj = np.ones_like(r_traj,dtype='int') * self.k

        rtg_raw_traj, J_tot, J_disc = reward_calc(r_raw_traj,self.gamma)

        return (s_traj, s_raw_traj, a_traj, r_traj, sp_traj, d_traj, 
            neglogp_traj, klinfo_traj, k_traj, rtg_raw_traj, 
            J_tot, J_disc, full)

    def generate_batch(self,env,actor):
        """Generates batch of trajectories."""
        traj_batch = 0
        self.steps_batch = 0

        while self.steps_batch < self.b_size:
            res = self._generate_traj(env,actor)
            
            (s_traj, s_raw_traj, a_traj, r_traj, sp_traj, d_traj, 
                neglogp_traj, klinfo_traj, k_traj, rtg_raw_traj, 
                J_tot, J_disc, full) = res

            # Store
            self.s_batch.append(s_traj)
            self.s_raw_batch.append(s_raw_traj)
            self.a_batch.append(a_traj)
            self.r_batch.append(r_traj)
            self.sp_batch.append(sp_traj)
            self.d_batch.append(d_traj)
            self.neglogp_batch.append(neglogp_traj)
            self.klinfo_batch.append(klinfo_traj)
            self.k_batch.append(k_traj)
            if full:
                self.rtg_raw_batch.append(rtg_raw_traj)
                self.J_tot_batch.append(J_tot)
                self.J_disc_batch.append(J_disc)
                self.traj_len_batch.append(len(r_traj))

            traj_batch += 1
        
        self.traj_total += traj_batch
        self.steps_total += self.steps_batch
    
    def get_log_info(self):
        """Returns dictionary of info for logging."""
        J_tot_ave = np.mean(self.J_tot_batch)
        J_disc_ave = np.mean(self.J_disc_batch)
        traj_len_ave = np.mean(self.traj_len_batch)
        log_info = {
            'J_tot':    J_tot_ave,
            'J_disc':   J_disc_ave,
            'traj':     self.traj_total,
            'steps':    self.steps_total,
            'traj_len': traj_len_ave
        }
        return log_info

    def get_update_info(self,actor,critic):
        """Returns data needed to calculate actor and critic updates."""
        if (self.noldpols > 0) and (len(self.s_buffer) > 0):
            s_all = [self.s_batch] + self.s_buffer
            a_all = [self.a_batch] + self.a_buffer
            neglogp_all = [self.neglogp_batch] + self.neglogp_buffer
            klinfo_all = [self.klinfo_batch] + self.klinfo_buffer

            k_all = [self.k_batch] + self.k_buffer
            M_active = len(self.s_buffer) + 1
            weights_active = self.weights[:M_active]
            weights_active = weights_active / np.sum(weights_active)
            weights_active = weights_active * M_active

            r_all = [self.r_batch] + self.r_buffer
            sp_all = [self.sp_batch] + self.sp_buffer
            d_all = [self.d_batch] + self.d_buffer

            adv_all, rtg_all = gae_all(s_all,a_all,sp_all,r_all,d_all,
                neglogp_all,self.gamma,self.lam,self.vtrace,
                self.is_trunc,actor,critic)

        else:
            s_all = [self.s_batch]
            a_all = [self.a_batch]
            neglogp_all = [self.neglogp_batch]
            klinfo_all = [self.klinfo_batch]

            k_all = [self.k_batch]
            weights_active = np.array([1.])

            r_all = [self.r_batch]
            sp_all = [self.sp_batch]
            d_all = [self.d_batch]

            # Uses GAE: no old data
            adv_all, rtg_all = gae_all(s_all,a_all,sp_all,r_all,d_all,
                neglogp_all,self.gamma,self.lam,False,
                self.is_trunc,actor,critic)
        
        s_all = aggregate_data(s_all)
        a_all = aggregate_data(a_all)
        adv_all = aggregate_data(adv_all)
        rtg_all = aggregate_data(rtg_all)
        neglogp_all = aggregate_data(neglogp_all)
        klinfo_all = aggregate_data(klinfo_all)

        age_all = self.k - aggregate_data(k_all)
        weights_all = weights_active[age_all].astype('float32')

        return (s_all, a_all, adv_all, rtg_all, neglogp_all, klinfo_all, 
            weights_all)

    def get_env_info(self):
        """Returns data used to update running normalization stats."""
        s_raw = aggregate_data([self.s_raw_batch])
        rtg_raw = aggregate_data([self.rtg_raw_batch])

        return s_raw, rtg_raw