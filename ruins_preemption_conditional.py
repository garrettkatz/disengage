import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import preemption_conditional as pc
from ruins_conditional import setup_mu, EpisodeRunner
change to cp/ll versions, make sure to close runners, and run failure rates first
if __name__ == "__main__":

    do_conform = True
    do_show = True

    # don't need gradients now
    tr.set_grad_enabled(False)

    params = {
        "basename": "ru_data/ru_cond",

        "train_rep": 1,    
        "train_obs_noise": .05,
        "calib_obs_noise": .0,
        "final_obs_noise": .0,
    
        "max_leadtime": 5,
        "num_updates": 5000,
        "learning_rate": 1e-2,
        "weight_decay": .1,
    
        "delta_ratio": .5,
        "calib_padding": 0, # this many outliers below tau
        "num_repetitions": 100, # this many repetitions to estimate preemption rates
    }

    if do_conform:
        run_calib = EpisodeRunner(params["calib_obs_noise"])
        run_final = EpisodeRunner(params["final_obs_noise"])
        pc.conform(params, setup_mu, run_calib, run_final)
        run_calib.close()
        run_final.close()

    if do_show:
        pc.show_results(params)


