import torch
import torch.nn as nn
import algorithm.helper as h
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class AWRNetwork(nn.Module):
    """Encoder and policy head for BC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self._encoder = h.enc(cfg)
        self._Q_encoder = h.enc(cfg)
        self._V_encoder = h.enc(cfg)
        self._pi_encoder = h.enc(cfg)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Qs = nn.ModuleList([h.q(cfg) for _ in range(cfg.num_q)])
        self._V = h.v(cfg)
        self.apply(h.orthogonal_init)     
        for m in [*self._Qs, self._V]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in self._Qs:
            h.set_requires_grad(m, enable)

    def track_v_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        if hasattr(self, '_V'):
            h.set_requires_grad(self._V, enable)

    def encode(self, obs):
        """Encodes an observation into its latent representation."""
        out = self._encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def V(self, z):
        """Predict state value (V)."""
        return self._V(z)

    def Q(self, z, a, return_type):
        """Predict state-action value (Q)."""
        assert return_type in {'min', 'avg', 'all'}
        x = torch.cat([z, a], dim=-1)

        if return_type == 'all':
            return torch.stack(list(q(x) for q in self._Qs), dim=0)

        idxs = np.random.choice(self.cfg.num_q, 2, replace=False)
        Q1, Q2 = self._Qs[idxs[0]](x), self._Qs[idxs[1]](x)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
    
class AWR():
    """Implementation of AWR learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = AWRNetwork(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()
        self.batch_size = cfg.batch_size

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, step=None):
        """Take an action. Uses either MPC or the learned policy, depending on the self.cfg.mpc flag."""
        if isinstance(obs, dict):
            obs = {k: torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0) for k, o in obs.items()}
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # z = self.model.encode(obs)
        zpi = self.model._pi_encoder(obs)
        a = self.model.pi(zpi, self.cfg.min_std * (not eval_mode)).squeeze(0)
        return a.cpu()
    
    @torch.no_grad()
    def _td_target(self, next_z, reward, mask):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_v = self.model.V(next_z)
        td_target = reward + self.cfg.discount * mask * next_v
        return td_target
    
    def update(self, replay_buffer, step, demo_buffer=None):
        if demo_buffer is not None:
            # Update oversampling ratio
            self.demo_batch_size = int(
                h.linear_schedule(self.cfg.demo_schedule, step) * self.batch_size
            )
            replay_buffer.cfg.batch_size = self.batch_size - self.demo_batch_size
            demo_buffer.cfg.batch_size = self.demo_batch_size
        else:
            self.demo_batch_size = 0
        # Sample from interaction dataset
        obs, next_obses, action, reward, mask, terminated, done, idxs, weights = replay_buffer.sample()
        action.squeeze_(0)
        next_obses.squeeze_(0)
        reward.squeeze_(0)
        mask.squeeze_(0)
        terminated.squeeze_(0)
        done.squeeze_(0)

        # Sample from demonstration dataset
        if self.demo_batch_size > 0:
            (demo_obs, demo_next_obses, demo_action, demo_reward, demo_mask, demo_terminated, demo_done, demo_idxs, demo_weights
             ) = demo_buffer.sample()
            demo_action.squeeze_(0)
            demo_next_obses.squeeze_(0)
            demo_reward.squeeze_(0)
            demo_mask.squeeze_(0)
            demo_terminated.squeeze_(0)
            demo_done.squeeze_(0)

            if isinstance(obs, dict):
                obs = {k: torch.cat([obs[k], demo_obs[k]]) for k in obs}
                next_obses = {k: torch.cat([next_obses[k], demo_next_obses[k]], dim=0) for k in next_obses}
            else:
                obs = torch.cat([obs, demo_obs])
                next_obses = torch.cat([next_obses, demo_next_obses], dim=0)
            action = torch.cat([action, demo_action], dim=0)
            reward = torch.cat([reward, demo_reward], dim=0)
            mask = torch.cat([mask, demo_mask], dim=0)
            terminated = torch.cat([terminated, demo_terminated], dim=0)
            done = torch.cat([done, demo_done], dim=0)

        horizon = self.cfg.horizon
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # z = self.model.encode(obs) # (batch, latent_dim)
        zq = self.model._Q_encoder(obs)
        zv = self.model._V_encoder(obs)
        zpi = self.model._pi_encoder(obs)
        # Compute Q and V learning targets
        with torch.no_grad():
            # next_z = self.model.encode(next_obses)
            next_zv = self.model_target._V_encoder(next_obses)
            stop = terminated if self.cfg.bootstrap_terminal else done
            td_targets = self._td_target(next_zv, reward, ~stop) # (batch, 1)
            v_target = self.model_target.Q(zq.detach(), action, return_type='min')
        
        value_info = {'Q': 0., 'V': 0.}
        qs = self.model.Q(zq, action, return_type='all') # (ensemble, batch, 1)
        value_info['Q'] = qs.mean().item()
        v = self.model.V(zv) # (batch, 1)
        value_info['V'] = v.mean().item()
        q_value_loss = F.mse_loss(qs, td_targets)
        self.expectile = h.linear_schedule(self.cfg.expectile, step)
        v_value_loss = h.l2_expectile(v_target - v, expectile=self.expectile, reduce=True)

        # Update policy with AWR
        self.optim.zero_grad(set_to_none=True)
        prediction = self.model.pi(zpi)
        log_probs = h.gaussian_logprob(prediction - action, 0)
        # compute the advantage    
        adv = (v_target - v).detach()

        awr_filter = self.cfg.awr_filter
        if self.cfg.switch_awr_filter:
            awr_filter = 'exp' if demo_buffer is None else  'indicator'
                
        if awr_filter == 'indicator':
            adv_weights = (adv > 0).float()
        elif awr_filter == 'exp':
            adv_weights = torch.exp(self.cfg.A_scaling * adv).clamp_max(100)
        pi_loss = -(adv_weights * log_probs).mean()
        # also compute the RMSE for logging purposes.
        mse = F.mse_loss(prediction.detach(), action)

        if step % self.cfg.update_freq == 0:
            # h.ema(self.model._encoder, self.model_target._encoder, self.cfg.tau)
            h.ema(self.model._Q_encoder, self.model_target._Q_encoder, self.cfg.tau)
            h.ema(self.model._V_encoder, self.model_target._V_encoder, self.cfg.tau)
            h.ema(self.model._pi_encoder, self.model_target._pi_encoder, self.cfg.tau)
            h.ema(self.model._Qs, self.model_target._Qs, self.cfg.tau)

        
        loss = q_value_loss + v_value_loss +  (1.0 * pi_loss)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False))
        self.optim.step()

        self.model.eval()
        metrics = {
            'loss': float(loss.item()),
            'Q_value_loss': float(q_value_loss.mean().item()),
            'V_value_loss': float(v_value_loss.mean().item()),
            'pi_loss': pi_loss.item(),
            'pi_rmse': mse.sqrt().item(),
            'grad_norm': float(grad_norm),
            'adv_weights_hist': adv_weights[:128, 0].cpu().numpy(),
            'adv_hist': adv[:128, 0].cpu().numpy(),
            'adv_pos_ratio': (adv > 0).float().mean().item(),
        }
        if self.demo_batch_size > 0:
            metrics['demo_adv_pos_ratio'] = (adv[:self.demo_batch_size] > 0).float().mean().item()
            metrics['online_adv_pos_ratio'] = (adv[self.demo_batch_size:] > 0).float().mean().item()
            
        for key in ["demo_batch_size", "expectile"]:
            if hasattr(self, key):
                metrics[key] = getattr(self, key)
        metrics.update(value_info)
        return metrics