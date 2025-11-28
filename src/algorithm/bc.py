import torch
import torch.nn as nn
import algorithm.helper as h
import torch.nn.functional as F


class BCNetwork(nn.Module):
    """Encoder and policy head for BC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self.apply(h.orthogonal_init)

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


class BC():
    """Implementation of BC learning + inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = BCNetwork(cfg).cuda()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.model.eval()
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
    def act(self, obs, t0=False, eval_mode=False, step=None, to_cpu=True):
        """Take an action. Uses either MPC or the learned policy, depending on the self.cfg.mpc flag."""
        def to_device(x):
            return x.cpu() if to_cpu else x

        if isinstance(obs, dict):
            obs = {k: torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0) for k, o in obs.items()}
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.encode(obs)
        a = self.model.pi(z, self.cfg.min_std * (not eval_mode)).squeeze(0)
        return to_device(a)

    def update(self, replay_buffer, step):
   
        self.demo_batch_size = 0
        # Sample from interaction dataset
        obs, next_obses, action, reward, mask, terminated, done, idxs, weights = replay_buffer.sample()
        horizon = self.cfg.horizon
        loss_mask = torch.ones_like(mask, device=self.device)
        for t in range(1, horizon):
            loss_mask[t] = loss_mask[t - 1] * (~done[t - 1])

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Update policy
        z = self.model.encode(obs)
        self.optim.zero_grad(set_to_none=True)
        info = {}
        prediction = self.model.pi(z)
        log_probs = h.gaussian_logprob(prediction - action, 0)
        pi_loss = -((log_probs).mean(dim=(1, 2))).mean()
        pi_loss.backward()
        # also compute the RMSE for logging purposes.
        mse = F.mse_loss(prediction.detach(), action)
        info['pi_rmse'] = mse.sqrt().cpu().item()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False))
        self.optim.step()
        info['pi_loss'] = pi_loss.item()
        info['grad_norm'] = grad_norm

        self.model.eval()
        metrics = {}
        for key in ["demo_batch_size", "expectile"]:
            if hasattr(self, key):
                metrics[key] = getattr(self, key)

        metrics.update(info)

        return metrics