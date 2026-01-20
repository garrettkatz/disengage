import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su
import preemption_conditional as pc
from cartpole_conditional import setup_mu, EpisodeRunner

if __name__ == "__main__":

    do_conform = True
    do_show = False

    # don't need gradients now
    tr.set_grad_enabled(False)

    delta_ratios = [.1, .25, .5]
    obs_noises = [.4, .5, .6]
    params = {
        "basename": "cp_data/cp_cond",

        # "load_mus": True, # whether to load pretrained mus or use random initial weights
        # "train_rep": 0,

        "train_obs_noise": .4,
        "calib_obs_noise": .4,
        "final_obs_noise": .4,

        "max_episode_length": 500,

        "max_leadtime": 25,
        "num_updates": 25000,
        "learning_rate": 1e-5,
        "weight_decay": .5,
    
        "calib_padding": 0, # this many outliers below tau
        "num_repetitions": 100, # this many repetitions to estimate preemption rates

    }

    if do_conform:

        run_calib = EpisodeRunner(params["calib_obs_noise"], params["max_episode_length"])
        run_final = EpisodeRunner(params["final_obs_noise"], params["max_episode_length"])

        for (load_mus, train_rep) in it.product((False, True), range(5)):
            for (calib_obs_noise, final_obs_noise) in it.combinations_with_replacement(obs_noises, 2):
                params["load_mus"] = load_mus
                params["train_rep"] = train_rep
                params["calib_obs_noise"] = calib_obs_noise
                params["final_obs_noise"] = final_obs_noise
                pc.conform(params, delta_ratios, setup_mu, run_calib, run_final)

    if do_show:
        import matplotlib.pyplot as pt
        # pc.show_results(params)

        # want to show: average false alarm rates with trained mu are lower than untrained
        false_alarms = np.empty((2, 5, params["num_repetitions"], len(delta_ratios), params["max_leadtime"]))
        for (load_mus, train_rep) in it.product((False, True), range(5)):

            basename = params["basename"]
            train_obs_noise = params["train_obs_noise"]
            calib_obs_noise = params["calib_obs_noise"]
            final_obs_noise = params["final_obs_noise"]
            max_leadtime = params["max_leadtime"]
            num_updates = params["num_updates"]
            learning_rate = params["learning_rate"]
            weight_decay = params["weight_decay"]
            results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
                f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

            # load preemption results
            with open(results_name, "rb") as f:
                (failrate, _, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)
        
            false_alarms[int(load_mus), train_rep, :, :, :] = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])

        pt.plot(false_alarms[0,:,:,0,:].mean(axis=1).mean(axis=0), 'ro-', label="random")
        pt.plot(false_alarms[1,:,:,0,:].mean(axis=1).mean(axis=0), 'bo-', label="pretrained")
        pt.xlabel("L")
        pt.ylabel("False Alarm Rate")
        pt.legend()
        pt.show()

