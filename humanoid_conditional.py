from line_profiler import profile # python -m kernprof -lvr thisfile.py
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
import humanoid_utils as hu
import train_mu_conditional as tm

def setup_mu():
    # MLP - same architecture as critic in PPO policy for reach task
    num_hidden = 256
    return tr.nn.Sequential(
        tr.nn.Linear(in_features=55, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
        tr.nn.Tanh(),
        tr.nn.Linear(in_features=num_hidden, out_features=1, bias=True),
    )

# early termination = failure
def failure_predicate(env, obs, reward, done, infos): return done

class EpisodeRunner:

    def __init__(self, obs_noise, max_episode_length):
        self.env, self.model = hu.load()
        self.max_episode_length = max_episode_length
        self.perturb_obs = tm.perturb_obs_factory(obs_noise)

    def __call__(self):
        return hu.run(self.env, self.model, self.max_episode_length, failure_predicate, self.perturb_obs, render=False)

    def close(self):
        pass

if __name__ == "__main__":

    do_sampling = False
    do_training = False
    do_show = True

    # parity with original policy training:
    # 2B timesteps in buffer and fed through network
    # 2B / 16384 steps per minibatch = 122070 miniupdates.

    for rep in range(5):
        print(f"\n\n ******* {rep=} ********\n\n")

        params = {
            "basename": "hu_data/hu_cond",
            "buf_rep": rep,
            "train_rep": rep,
    
            "obs_noise": .1,
            "max_episode_length": 500,
            "total_timesteps": 200_000,
    
            "max_leadtime": 25,
            "num_updates": 10_400,
            "train_batch_size": 1000,
            "valid_batch_size": 10_000,
    
            "learning_rate": 1e-4,
            "weight_decay": 10.,
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
            tm.show_results(params, setup_mu, [1, 2, 5, 10, 25])
            # tm.show_results(params, setup_mu, [1])

