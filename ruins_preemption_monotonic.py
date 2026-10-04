import itertools as it
import pickle as pk
import numpy as np
import torch as tr
from time import perf_counter
import preemption_monotonic as pm
from ruins_ratio import EpisodeRunner
from ruins_monotonic import setup_mu_factory

if __name__ == "__main__":

    do_conform = True
    do_show = True

    # don't need gradients now
    tr.set_grad_enabled(False)

    delta_ratios = [.1, .25, .5]
    obs_noises = [.05]#, .25, .5]
    num_train_reps = 5

    params = {
        "basename": "ru_data/ru_mono",
        "buf_basename": "ru_data/ru_cond",

        # "load_mus": True, # whether to load pretrained mus or use random initial weights
        # "train_rep": 0,

        "train_obs_noise": obs_noises[0],
        "calib_obs_noise": obs_noises[0],
        "final_obs_noise": obs_noises[0],

        "max_episode_length": 240,

        "max_leadtime": 12,
        "num_updates": 10_400,
        "learning_rate": 5e-3,
        "weight_decay": .2,

        "calib_padding": 0, # this many outliers below tau
        "num_repetitions": 100, # this many repetitions to estimate preemption rates

    }

    setup_mu = setup_mu_factory(params["max_episode_length"])

    if do_conform:

        start_time = perf_counter()
        pm.conform_experiments(params, obs_noises, delta_ratios, num_train_reps, setup_mu, EpisodeRunner)
        print(f"Experiments took {perf_counter() - start_time}s")

    if do_show:
        pm.show_results(params, obs_noises, delta_ratios, num_train_reps)

