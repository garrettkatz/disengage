from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import sb_utils as su
import train_mu_conditional as tm

def setup_mu():
    # MLP - same architecture as critic
    num_hidden = 256
    return tr.nn.Sequential(
        tr.nn.Linear(in_features=8, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )

def failure_predicate(env, obs, reward, done, infos):
    # reward of -100 signals crash
    return bool(reward[0] == -100)

class EpisodeRunner:

    def __init__(self, obs_noise, max_episode_length):
        self.env, self.model = su.load("LunarLander-v3", "a2c")
        self.max_episode_length = max_episode_length
        self.perturb_obs = tm.perturb_obs_factory(obs_noise)

    def __call__(self):
        return su.run(self.env, self.model, self.max_episode_length, failure_predicate, perturb_obs=self.perturb_obs)

if __name__ == "__main__":

    do_sampling = False
    do_training = False
    do_show = True

    # parity with original policy training:
    # 200K timesteps in buffer (~ 200 rollouts)
    # 5000 updates * 40 batch size = 200,000 states fed through network

    for rep in range(5):
        print(f"\n\n ******* {rep=} ********\n\n")

        params = {
            "basename": "ll_data/ll_cond",
            "buf_rep": rep,
            "train_rep": rep,
    
            "obs_noise": .1,
            "max_episode_length": 1000,
            "total_timesteps": 200_000,
    
            "max_leadtime": 50,
            "num_updates": 5000,
            "train_batch_size": 40,
            "valid_batch_size": 20_000, # ~20 rollouts
    
            "learning_rate": 1e-3,
            "weight_decay": .5,
            "train_fraction": .8, # fraction of rollouts used for training
    
            "report_period": None,
            "checkpoint_period": 100,
            "bucket_size": 10, # buckets for learning curve trendlines
        }
    
        if do_sampling:
            tm.collect_data(params, EpisodeRunner(params["obs_noise"], params["max_episode_length"]))
    
        if do_training:
            tm.train(params, setup_mu)
    
        if do_show:
            tm.show_results(params, setup_mu, [1, 5, 10, 25, 50])

