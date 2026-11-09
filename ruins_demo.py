import numpy as np
import torch as tr
import ruins_utils as ru
import train_mu_conditional as tm
from ruins_conditional import setup_mu

render = True

obs_noise = .05

policy_checkpoint_name = "41652_ppo_drone"
env, model = ru.load(policy_checkpoint_name, render=render)
perturb_obs = tm.perturb_obs_factory(obs_noise)

mu = setup_mu()
tau = .65

# load pretrained state dicts
train_name = f"ru_data/ru_cond_train_rep0_on0.05_lr{5e-3}_wd{.2}_lt{12}_nu{10_400}_trained.pt"
(_, _, _, mu_state_dicts) = tr.load(train_name, weights_only=True)
mu.load_state_dict(mu_state_dicts[1])

_ = ru.run(env, model, perturb_obs, render=render, preemptor = (mu, tau))

input('..')

