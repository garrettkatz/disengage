from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import ruins_utils as ru
import train_mu_ratio as tm

def setup_mu():
    # MLP - same architecture as critic
    num_hidden = 64
    return tr.nn.Sequential(
        tr.nn.Linear(in_features=12, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )

class EpisodeRunner:

    def __init__(self, obs_noise, max_episode_length=None):
        # episode length parameter here for consistency with other envs, but ignored and always 240 for ruins
        policy_checkpoint_name = "41652_ppo_drone"
        self.env, self.model = ru.load(policy_checkpoint_name, render=False)
        self.perturb_obs = tm.perturb_obs_factory(obs_noise)

    def __call__(self):
        failure, ep_rew, ep_len, (observations, dep_imgs), actions = ru.run(self.env, self.model, self.perturb_obs, render=False)
        return failure, ep_rew, ep_len, observations, actions

    def close(self):
        self.env.close()

if __name__ == "__main__":

    do_training = False
    do_show = True

    # parity with original policy training:
    # 8_136_000 timesteps sampled and also fed through network (no off-policy replay buffer)
    # update every four episodes, so (41652 / 4 = 10413) updates, batch size <= 4 episodes * 240 max episode length
    # effective average batch size is = 8_136_000 / 10413 = 781.3

    for rep in range(1,5):
        print(f"\n\n ******* {rep=} ********\n\n")

        params = {
            "basename": "ru_data/ru_ratio",
            "buf_basename": "ru_data/ru_cond",
            "buf_rep": rep,
            "train_rep": rep,

            "obs_noise": .05,
            "max_episode_length": 240,
            "total_timesteps": 200_000,

            "max_leadtime": 12,
            "num_updates": 10_400,
            "train_batch_size": 781,
            "valid_batch_size": 20_000,

            "learning_rate": 5e-3,
            "weight_decay": .2,
            "train_fraction": .8, # fraction of rollouts used for training

            "report_period": None,
            "checkpoint_period": 100,
            "bucket_size": 10, # buckets for learning curve trendlines
        }

        if do_training:
            tm.train(params, setup_mu)

        if do_show:
            tm.show_results(params, setup_mu, [-1] + list(range(1,13)))

