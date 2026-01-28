import itertools as it
import pickle as pk
import numpy as np
import torch as tr
from time import perf_counter
import preemption_conditional as pc
from ruins_conditional import setup_mu, EpisodeRunner

if __name__ == "__main__":

    do_conform = False
    do_show = True

    # don't need gradients now
    tr.set_grad_enabled(False)

    delta_ratios = [.1, .25, .5]
    # obs_noises = [.05, .4, .6]
    obs_noises = [.05, .25, .5]
    num_train_reps = 5

    params = {
        "basename": "ru_data/ru_cond",

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

    if do_conform:

        start_time = perf_counter()
        pc.conform_experiments(params, obs_noises, delta_ratios, num_train_reps, setup_mu, EpisodeRunner)
        print(f"Experiments took {perf_counter() - start_time}s")

    if do_show:
        pc.show_results(params, obs_noises, delta_ratios, num_train_reps)

    # if do_conform:

    #     for (calib_obs_noise, final_obs_noise) in it.combinations_with_replacement(obs_noises, 2):
    #         run_calib = EpisodeRunner(calib_obs_noise, params["max_episode_length"])
    #         run_final = EpisodeRunner(final_obs_noise, params["max_episode_length"])

    #         for (load_mus, train_rep) in it.product((False, True), range(5)):
    #             params["load_mus"] = load_mus
    #             params["train_rep"] = train_rep
    #             params["calib_obs_noise"] = calib_obs_noise
    #             params["final_obs_noise"] = final_obs_noise
    #             pc.conform(params, delta_ratios, setup_mu, run_calib, run_final)

    #         run_calib.close()
    #         run_final.close()

    # if do_show:
    #     import matplotlib.pyplot as pt
    #     # pc.show_results(params)

    #     # want to show: some robustness of preemption rate under distribution shift
    #     for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

    #         Ls = np.arange(params["max_leadtime"])+1
    #         unpreempted = np.empty((2, 5, params["num_repetitions"], len(delta_ratios), params["max_leadtime"]))
    #         for (load_mus, train_rep) in it.product((False, True), range(5)):
    
    #             basename = params["basename"]
    #             train_obs_noise = params["train_obs_noise"]
    #             # calib_obs_noise = params["calib_obs_noise"]
    #             # final_obs_noise = params["final_obs_noise"]
    #             max_leadtime = params["max_leadtime"]
    #             num_updates = params["num_updates"]
    #             learning_rate = params["learning_rate"]
    #             weight_decay = params["weight_decay"]
    #             results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
    #                 f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"
    
    #             # load preemption results
    #             with open(results_name, "rb") as f:
    #                 (failrate, _, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)

    #             # unpreempted if failed and first alarm came at or later than max(1, fail time - L)
    #             unpreempted[int(load_mus), train_rep, :, :, :] = deploy_failures[:,None,None] & (alarm_times >= np.maximum(1, deploy_durations[:,None,None] - Ls))
    
    #         pt.subplot(2,3,sp+1)
    #         rand_mean = unpreempted[0,:,:,0,:].mean(axis=1).mean(axis=0)
    #         rand_stdv = unpreempted[0,:,:,0,:].mean(axis=1).std(axis=0)
    #         pret_mean = unpreempted[1,:,:,0,:].mean(axis=1).mean(axis=0)
    #         pret_stdv = unpreempted[1,:,:,0,:].mean(axis=1).std(axis=0)
    #         pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
    #         pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
    #         pt.plot(Ls, rand_mean, 'ro:', label="random")
    #         pt.plot(Ls, pret_mean, 'b^-', label="pretrained")
    #         pt.plot(Ls, [deltas[0]]*params["max_leadtime"], 'k--', label="delta")
    #         # pt.xlabel("L")
    #         # pt.ylabel("False Alarm Rate")
    #         pt.title(f"calib={calib_obs_noise}, deploy={final_obs_noise}")

    #     pt.legend()
    #     pt.gcf().supxlabel("L")
    #     pt.gcf().supylabel("Unpreempted Failure Rate")
    #     pt.tight_layout()
    #     pt.show()

    #     # want to show: average false alarm rates with trained mu are lower than untrained
    #     for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

    #         false_alarms = np.empty((2, 5, params["num_repetitions"], len(delta_ratios), params["max_leadtime"]))
    #         for (load_mus, train_rep) in it.product((False, True), range(5)):
    
    #             basename = params["basename"]
    #             train_obs_noise = params["train_obs_noise"]
    #             # calib_obs_noise = params["calib_obs_noise"]
    #             # final_obs_noise = params["final_obs_noise"]
    #             max_leadtime = params["max_leadtime"]
    #             num_updates = params["num_updates"]
    #             learning_rate = params["learning_rate"]
    #             weight_decay = params["weight_decay"]
    #             results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
    #                 f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"
    
    #             # load preemption results
    #             with open(results_name, "rb") as f:
    #                 (failrate, _, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)
            
    #             false_alarms[int(load_mus), train_rep, :, :, :] = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
    
    #         pt.subplot(2,3,sp+1)
    #         rand_mean = false_alarms[0,:,:,0,:].mean(axis=1).mean(axis=0)
    #         rand_stdv = false_alarms[0,:,:,0,:].mean(axis=1).std(axis=0)
    #         pret_mean = false_alarms[1,:,:,0,:].mean(axis=1).mean(axis=0)
    #         pret_stdv = false_alarms[1,:,:,0,:].mean(axis=1).std(axis=0)
    #         Ls = np.arange(len(rand_mean))+1
    #         pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
    #         pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
    #         pt.plot(Ls, rand_mean, 'ro:', label="random")
    #         pt.plot(Ls, pret_mean, 'b^-', label="pretrained")
    #         # pt.xlabel("L")
    #         # pt.ylabel("False Alarm Rate")
    #         pt.title(f"calib={calib_obs_noise}, deploy={final_obs_noise}")

    #     pt.legend()
    #     pt.gcf().supxlabel("L")
    #     pt.gcf().supylabel("False Alarm Rate")
    #     pt.tight_layout()
    #     pt.show()


