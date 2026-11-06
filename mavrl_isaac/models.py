"""Reuse the unmodified upstream VAE; port the policy/LSTM modules verbatim in shape.
See docs/SOURCE_AUDIT.md for mapping and modern-runtime compatibility changes.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from torch.distributions import Normal
spec=importlib.util.spec_from_file_location('mavrl_original_vae',Path(__file__).resolve().parents[1]/'upstream/mavrl/models/vae.py')
upstream=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
VAE=upstream.VAE

def orthogonal(module,gain=2**0.5):
    if isinstance(module,(nn.Linear,nn.Conv2d)):
        nn.init.orthogonal_(module.weight,gain)
        if module.bias is not None:nn.init.zeros_(module.bias)

class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=upstream.Encoder(1,64)
        self.encoder.apply(orthogonal)
        self.lstm=nn.LSTM(64,256,num_layers=1)
        self.critic_lstm=None
        self.mu_linear=nn.Linear(256,192)
        # Upstream LSTM reconstruction decoder is randomly initialized, shared
        # across the 3 targets, and optimized with the LSTM; not the VAE decoder.
        self.decoder=upstream.Decoder(1,64)

    def zero_state(self,n):
        return tuple(torch.zeros(1,n,256,device=next(self.parameters()).device) for _ in range(2))

    def enable_separate_critic(self):
        # Official policy reload omits recurrent constructor flags and therefore
        # uses enable_critic_lstm=True, unlike initial PPO's shared LSTM.
        if self.critic_lstm is None:
            self.critic_lstm=nn.LSTM(64,256,num_layers=1).to(next(self.parameters()).device)
        return self

    @torch.no_grad()
    def step_both(self,image,actor_state,critic_state,episode_starts):
        x=torch.as_tensor(image,device=next(self.parameters()).device).float()/255.
        mu,_=self.encoder(x)
        mask=(~torch.as_tensor(episode_starts,device=x.device).bool()).float()[None,:,None]
        pi,actor_state=self.lstm(mu[None],tuple(s*mask for s in actor_state))
        if self.critic_lstm is None:return pi[0],pi[0],actor_state,actor_state
        vf,critic_state=self.critic_lstm(mu[None],tuple(s*mask for s in critic_state))
        return pi[0],vf[0],actor_state,critic_state

    @torch.no_grad()
    def step(self,image,state,episode_starts,reset_memory=False):
        x=torch.as_tensor(image,device=next(self.parameters()).device).float()/255.
        mu,_=self.encoder(x)
        starts=torch.as_tensor(episode_starts,device=x.device).bool()
        if reset_memory:starts=torch.ones_like(starts)
        mask=(~starts).float()[None,:,None]
        output,state=self.lstm(mu[None],tuple(s*mask for s in state))
        return output[0],state

    def freeze(self):
        self.eval().requires_grad_(False)
        return self

class ActorCritic(nn.Module):
    def __init__(self,log_std_init=-0.5):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(263,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.critic=nn.Sequential(nn.Linear(263,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU())
        self.action_net=nn.Sequential(nn.Linear(256,4),nn.Tanh())
        self.value_net=nn.Linear(512,1)
        self.log_std=nn.Parameter(torch.full((4,),log_std_init))
        self.actor.apply(orthogonal);self.critic.apply(orthogonal)
        orthogonal(self.action_net[0],0.01);orthogonal(self.value_net,1.)

    def distribution_value(self,x,critic_x=None):
        # Upstream tanh bounds the MEAN, then Gaussian samples are clipped for
        # environment execution. PPO stores the UNCLIPPED sample/log probability.
        return Normal(self.action_net(self.actor(x)),self.log_std.exp()),self.value_net(self.critic(x if critic_x is None else critic_x)).squeeze(-1)

    def act(self,x,deterministic=False,critic_x=None):
        dist,value=self.distribution_value(x,critic_x)
        action=dist.mean if deterministic else dist.sample()
        return action,value,dist.log_prob(action).sum(-1)
