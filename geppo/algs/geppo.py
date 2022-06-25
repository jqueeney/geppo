import numpy as np
import tensorflow as tf

from geppo.algs.base_alg import BaseAlg

class GePPO(BaseAlg):
    """Algorithm class for GePPO. PPO is a special case."""

    def __init__(self,seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes GePPO class. See BaseAlg for details."""

        super(GePPO,self).__init__(seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)

        self._ac_setup()

    def _ac_setup(self):
        """Sets up actor and critic kwargs as class attributes."""
        self.critic_lr = self.ac_kwargs['critic_lr']        
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)

        self.actor_lr = self.ac_kwargs['actor_lr']
        if self.ac_kwargs['scaleinitlr_dim']:
            self.actor_lr = self.actor_lr / self.env.action_dim
        if self.ac_kwargs['scaleinitlr_eps']:
            self.actor_lr = self.actor_lr * self.ac_kwargs['eps_mult']
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.actor_lr)

        self.update_it = self.ac_kwargs['update_it']
        self.nminibatch = self.ac_kwargs['nminibatch']
        self.max_grad_norm = self.ac_kwargs['max_grad_norm']

        self.adv_center = self.ac_kwargs['adv_center']
        self.adv_scale = self.ac_kwargs['adv_scale']
        self.adv_clip = self.ac_kwargs['adv_clip']
        
        self.eps_ppo = self.ac_kwargs['eps_ppo']
        self.eps = self.eps_ppo * self.ac_kwargs['eps_mult']
        self.eps_vary = self.ac_kwargs['eps_vary']
        
        self.adapt_lr = self.ac_kwargs['adapt_lr']
        self.adapt_factor = self.ac_kwargs['adapt_factor']
        self.adapt_minthresh = self.ac_kwargs['adapt_minthresh']
        self.adapt_maxthresh = self.ac_kwargs['adapt_maxthresh']

        self.early_stop = self.ac_kwargs['early_stop']
    
    def _get_neg_pg(self,s_active,a_active,adv_active,neglogp_old_active,
        weights_active):
        """Calculates negative gradient of policy objective.

        Args:
            s_active (np.ndarray): states
            a_active (np.ndarray): actions
            adv_active (np.ndarray): advantages
            neglogp_old_active (np.ndarray): negative log probabilities
            weights_active (np.ndarray): policy weights
        
        Returns:
            Policy objective and negative gradient w.r.t. policy parameters.
        """

        neglogp_pik_active = self.actor.neglogp_pik(s_active,a_active)
        offpol_ratio = tf.exp(neglogp_old_active - neglogp_pik_active)

        adv_mean = (np.mean(offpol_ratio * weights_active * adv_active) / 
            np.mean(offpol_ratio * weights_active))
        adv_std = np.std(offpol_ratio * weights_active * adv_active) + 1e-8

        if self.adv_center:
            adv_active = adv_active - adv_mean
        if self.adv_scale:
            adv_active = adv_active / adv_std
        if self.adv_clip:
            adv_active = np.clip(adv_active,-self.adv_clip,self.adv_clip)

        with tf.GradientTape() as tape:
            neglogp_cur_active = self.actor.neglogp(s_active,a_active)
            ratio = tf.exp(neglogp_old_active - neglogp_cur_active)
            ratio_clip = tf.clip_by_value(
                ratio,offpol_ratio-self.eps,offpol_ratio+self.eps)
            
            pg_loss_surr = ratio * adv_active * -1
            pg_loss_clip = ratio_clip * adv_active * -1
            pg_loss = tf.reduce_mean(
                tf.maximum(pg_loss_surr,pg_loss_clip)*weights_active)

        neg_pg = tape.gradient(pg_loss,self.actor.trainable)
        
        return pg_loss, neg_pg

    def _update(self):
        """Updates actor and critic."""
        data_all = self.runner.get_update_info(self.actor,self.critic)
        (s_all, a_all, adv_all, rtg_all, neglogp_old_all, kl_info_all, 
            weights_all) = data_all
        n_samples = s_all.shape[0]
        n_batch = int(n_samples / self.nminibatch)

        ent = tf.reduce_mean(weights_all * self.actor.entropy(s_all))
        kl_info_ref = self.actor.get_kl_info(s_all)

        pg_loss_all = 0
        pg_norm_all_pre = 0
        pg_norm_all = 0
        v_loss_all = 0
        vg_norm_all = 0
        
        if self.eps_vary:
            neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
            offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)
            eps_old = tf.reduce_mean(weights_all * tf.abs(offpol_ratio-1.))
            self.eps = np.maximum(self.eps_ppo - eps_old,0.0)

        # Minibatch update loop for actor and critic
        for sweep_it in range(self.update_it):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,n_batch)[1:]

            batches = np.array_split(idx,sections)
            if (n_samples % n_batch != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                # Active data
                s_active = s_all[batch_idx]
                a_active = a_all[batch_idx]
                adv_active = adv_all[batch_idx]
                rtg_active = rtg_all[batch_idx]
                neglogp_old_active = neglogp_old_all[batch_idx]
                weights_active = weights_all[batch_idx]

                # Critic update
                with tf.GradientTape() as tape:
                    V = self.critic.value(s_active)
                    v_loss = 0.5 * tf.reduce_mean(
                        weights_active * tf.square(rtg_active - V))
                
                vg = tape.gradient(v_loss,self.critic.trainable)
                self.critic_optimizer.apply_gradients(
                    zip(vg,self.critic.trainable))

                v_loss_all += v_loss
                vg_norm_all += tf.linalg.global_norm(vg)

                # Actor update
                pg_loss, neg_pg = self._get_neg_pg(s_active,a_active,adv_active,
                    neglogp_old_active,weights_active)
                
                if self.max_grad_norm is not None:
                    neg_pg, pg_norm_pre = tf.clip_by_global_norm(
                        neg_pg,self.max_grad_norm)
                else:
                    pg_norm_pre = tf.linalg.global_norm(neg_pg)
                
                if self.eps > 0:
                    self.actor_optimizer.apply_gradients(
                        zip(neg_pg,self.actor.trainable))
                
                pg_loss_all += pg_loss
                pg_norm_all_pre += pg_norm_pre
                pg_norm_all += tf.linalg.global_norm(neg_pg)

            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            clip_center = tf.exp(neglogp_old_all - neglogp_pik_all)
            ratio_diff = tf.abs(ratio - clip_center)
            
            tv = 0.5 * tf.reduce_mean(weights_all * ratio_diff)
            pen = 0.5 * tf.reduce_mean(weights_all * tf.abs(ratio-1.))

            # Early stopping
            if self.early_stop and (tv > (0.5*self.eps)):
                break

        pg_loss_ave = pg_loss_all.numpy() / ((sweep_it+1)*len(batches))
        pg_norm_ave_pre = pg_norm_all_pre.numpy() / ((sweep_it+1)*len(batches))
        pg_norm_ave = pg_norm_all.numpy() / ((sweep_it+1)*len(batches))

        pg_loss_final, _ = self._get_neg_pg(s_all,a_all,adv_all,neglogp_old_all,
            weights_all)

        v_loss_ave = v_loss_all.numpy() / ((sweep_it+1)*len(batches))
        vg_norm_ave = vg_norm_all.numpy() / ((sweep_it+1)*len(batches))

        log_critic = {
            'critic_loss':      v_loss_ave,
            'critic_grad_norm': vg_norm_ave
        }
        self.logger.log_train(log_critic)
        
        kl = tf.reduce_mean(weights_all * self.actor.kl(s_all,kl_info_ref))
        pen_kl = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all))

        kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_ref,direction='reverse'))
        pen_kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all,direction='reverse'))     
        
        log_actor = {
            'pg_loss':              pg_loss_ave,
            'pg_loss_final':        pg_loss_final.numpy(),
            'pg_norm_pre':          pg_norm_ave_pre,
            'pg_norm':              pg_norm_ave,
            'ent':                  ent.numpy(),
            'tv':                   tv.numpy(),
            'kl':                   kl.numpy(),
            'kl_reverse':           kl_reverse.numpy(),
            'penalty':              pen.numpy(),
            'penalty_kl':           pen_kl.numpy(),
            'penalty_kl_reverse':   pen_kl_reverse.numpy(),
            'outside_clip':         np.mean(ratio_diff > self.eps),
            'actor_sweeps':         sweep_it + 1,
            'actor_lr':             self.actor_optimizer.learning_rate.numpy(),
            'eps':                  self.eps,
        }
        self.logger.log_train(log_actor)
        
        self.actor.update_pik_weights()

        # Adapt learning rate
        if self.adapt_lr:
            if tv > (self.adapt_maxthresh * (0.5*self.eps)):
                lr_new = (self.actor_optimizer.learning_rate.numpy() / 
                    (1+self.adapt_factor))
                self.actor_optimizer.learning_rate.assign(lr_new)
            elif tv < (self.adapt_minthresh * (0.5*self.eps)):
                lr_new = (self.actor_optimizer.learning_rate.numpy() * 
                    (1+self.adapt_factor))
                self.actor_optimizer.learning_rate.assign(lr_new)