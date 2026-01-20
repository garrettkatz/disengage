import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import sb_utils as su
import preemption_conditional as pc
from lunar_lander_conditional import setup_mu, EpisodeRunner

if __name__ == "__main__":

    do_conform = True
    do_show = False

    # don't need gradients now
    tr.set_grad_enabled(False)

    delta_ratios = [.1, .25, .5]
    obs_noises = [.1, .2, .3]
    params = {
        "basename": "ll_data/ll_cond",

        # "load_mus": True, # whether to load pretrained mus or use random initial weights
        # "train_rep": 0,

        "train_obs_noise": obs_noises[0],
        "calib_obs_noise": obs_noises[0],
        "final_obs_noise": obs_noises[0],

        "max_episode_length": 1000,

        "max_leadtime": 50,
        "num_updates": 5000,
        "learning_rate": 1e-3,
        "weight_decay": .5,
    
        "calib_padding": 0, # this many outliers below tau
        "num_repetitions": 100, # this many repetitions to estimate preemption rates

    }

    if do_conform:

        for (calib_obs_noise, final_obs_noise) in it.combinations_with_replacement(obs_noises, 2):
            run_calib = EpisodeRunner(calib_obs_noise, params["max_episode_length"])
            run_final = EpisodeRunner(final_obs_noise, params["max_episode_length"])

            for (load_mus, train_rep) in it.product((False, True), range(5)):
                params["load_mus"] = load_mus
                params["train_rep"] = train_rep
                params["calib_obs_noise"] = calib_obs_noise
                params["final_obs_noise"] = final_obs_noise
                pc.conform(params, delta_ratios, setup_mu, run_calib, run_final)

            run_calib.close()
            run_final.close()

    if do_show:
        import matplotlib.pyplot as pt
        # pc.show_results(params)

        # want to show: average false alarm rates with trained mu are lower than untrained

        for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

            false_alarms = np.empty((2, 5, params["num_repetitions"], len(delta_ratios), params["max_leadtime"]))
            for (load_mus, train_rep) in it.product((False, True), range(5)):
    
                basename = params["basename"]
                train_obs_noise = params["train_obs_noise"]
                # calib_obs_noise = params["calib_obs_noise"]
                # final_obs_noise = params["final_obs_noise"]
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
    
            pt.subplot(2,3,sp+1)
            rand_mean = false_alarms[0,:,:,0,:].mean(axis=1).mean(axis=0)
            rand_stdv = false_alarms[0,:,:,0,:].mean(axis=1).std(axis=0)
            pret_mean = false_alarms[1,:,:,0,:].mean(axis=1).mean(axis=0)
            pret_stdv = false_alarms[1,:,:,0,:].mean(axis=1).std(axis=0)
            Ls = np.arange(len(rand_mean))+1
            pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
            pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
            pt.plot(Ls, rand_mean, 'ro:', label="random")
            pt.plot(Ls, pret_mean, 'b^-', label="pretrained")
            # pt.xlabel("L")
            # pt.ylabel("False Alarm Rate")
            pt.title(f"calib={calib_obs_noise}, deploy={final_obs_noise}")

        pt.legend()
        pt.gcf().supxlabel("L")
        pt.gcf().supylabel("False Alarm Rate")
        pt.tight_layout()
        pt.show()


# def main():

#     do_reps = True
#     do_show = True

#     env_name = "LunarLander-v3"
#     alg_name = "a2c"
#     max_episode_length = 1000
#     train_obs_noise = .1
#     basename = "ll_data/ll_cond"

#     max_leadtime = 10
#     train_batch_size = 40
#     learning_rate = 1e-3
#     train_basename = f"{basename}_train_on{train_obs_noise}_lr{learning_rate}_lt{max_leadtime}"

#     mu_updates = [1]
#     delta_ratios = [.5]
#     calib_padding = 0 # this many outliers below tau
#     calib_obs_noise = .1
#     final_obs_noise = .1

#     num_repetitions = 50
#     results_basename = f"{basename}_conform_on{train_obs_noise}_lr{learning_rate}_lt{max_leadtime}"

#     # don't need gradients now
#     tr.set_grad_enabled(False)

#     if do_reps:

#         # load environment and blackbox policy
#         env, model = su.load(env_name, alg_name)

#         for (num_updates, delta_ratio) in it.product(mu_updates, delta_ratios):

#             # load failrate and mus
#             (failrate, _, _, mu_state_dicts) = tr.load(f"{train_basename}_nu{num_updates}_trained.pt", weights_only=True)
#             mu = {}
#             for L in range(1, max_leadtime + 1):
#                 mu[L] = setup_mu()
#                 mu[L].load_state_dict(mu_state_dicts[L])

#             # setup calibration parameters delta
#             delta = delta_ratio * failrate.item()
#             num_calibration = int(np.ceil((1 - delta + calib_padding) / delta)) # ensure non-infinite tau

#             deploy_failures = np.empty(num_repetitions, dtype=bool)
#             deploy_durations = np.empty(num_repetitions, dtype=int)
#             alarm_times = np.empty((num_repetitions, max_leadtime), dtype=int) # episode length signifies no alarm
#             for rep in range(num_repetitions):

#                 # collect calibration episodes
#                 calib_observations, calib_failure = [], []
#                 for episode in range(num_calibration):
#                     failure, _, _, observations, _ = su.run(env, model, max_episode_length, failure_predicate, perturb_obs=perturb_obs_factory(calib_obs_noise))
#                     calib_observations.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))
#                     calib_failure.append(failure)
    
#                 # final episode
#                 final_failure, _, _, final_observations, _ = su.run(env, model, max_episode_length, failure_predicate, perturb_obs=perturb_obs_factory(final_obs_noise))
#                 final_observations = tr.tensor(np.concatenate(final_observations, axis=0)).to(tr.float32)

#                 # conformal method for each L
#                 for L in range(1, max_leadtime + 1):
    
#                     # calibration stats
#                     calib_predictions = [mu[L](obs).squeeze() for obs in calib_observations]
#                     z = [(preds[:max(1,len(preds)-L)].max().item() if fail else np.inf) for (preds, fail) in zip(calib_predictions, calib_failure)]
                    
#                     # set tau
#                     z.sort(reverse=True)
#                     tau = z[int(np.ceil((num_calibration + 1)*(1 - delta)))-1]
    
#                     # check for alarm on final episode
#                     preds = mu[L](final_observations).squeeze()
#                     alarms = (preds[:max(1, len(preds)-L)] > tau).numpy()
#                     if alarms.any():
#                         alarm_times[rep, L-1] = alarms.argmax()
#                     else:
#                         alarm_times[rep, L-1] = len(final_observations)

#                 deploy_failures[rep] = final_failure
#                 deploy_durations[rep] = len(final_observations)
#                 print(f" {failrate=}, {num_updates=}, {delta=}, {num_calibration=}, {rep=}, {final_failure=}, dur={len(final_observations)}, alarms at:", alarm_times[rep])

#             print(f"{failrate=}, {num_updates=}, {delta=}, {num_calibration=}:")
#             sound_alarms =  deploy_failures[:,None] & (alarm_times < deploy_durations[:,None])
#             false_alarms = ~deploy_failures[:,None] & (alarm_times < deploy_durations[:,None])
#             unpreempted = deploy_failures[:,None] & (alarm_times==deploy_durations[:,None])
#             print(f" unpreempted rates: {unpreempted.mean(axis=0)}")
#             print(f" false alarm rates: {false_alarms.mean(axis=0)}")

#             with open(f"{results_basename}_nu{num_updates}_dr{delta_ratio}", "wb") as f:
#                 pk.dump((failrate, delta, num_calibration, deploy_failures, deploy_durations, alarm_times), f)

#     if do_show:

#         import matplotlib.pyplot as pt

#         for (num_updates, delta_ratio) in it.product(mu_updates, delta_ratios):

#             # load preemption results
#             with open(f"{results_basename}_nu{num_updates}_dr{delta_ratio}", "rb") as f:
#                 (failrate, delta, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)

#             print(f"{failrate=}, {num_updates=}, {delta=}, {num_calibration=}:")
#             sound_alarms =  deploy_failures[:,None] & (alarm_times < deploy_durations[:,None])
#             false_alarms = ~deploy_failures[:,None] & (alarm_times < deploy_durations[:,None])
#             unpreempted = deploy_failures[:,None] & (alarm_times==deploy_durations[:,None])
#             print(f" unpreempted rates: {unpreempted.mean(axis=0).round(3)}")
#             print(f" false alarm rates: {false_alarms.mean(axis=0).round(3)}")

#             pt.subplot(1,3,1)
#             pt.plot(np.arange(1, max_leadtime+1), false_alarms.mean(axis=0), 'bx-', label="false alarms")
#             pt.plot(np.arange(1, max_leadtime+1), unpreempted.mean(axis=0), 'ro-', label="unpreempted failures")
#             pt.plot(np.arange(1, max_leadtime+1), [delta]*max_leadtime, ':', color=(.5,)*3, label="delta")
#             pt.xlabel("L")
#             pt.ylabel("Rate")
#             pt.legend()

#             pt.subplot(1,3,2)
#             for L in range(max_leadtime):
#                 quiet_time = alarm_times[~deploy_failures,L] / deploy_durations[~deploy_failures]
#                 pt.plot([L+1]*len(quiet_time), quiet_time, 'k.')
#             pt.xlim([-1, max_leadtime+1])
#             pt.xlabel("L")
#             pt.ylabel("Alarm-free portion of non-failures")

#             pt.subplot(1,3,3)
#             for L in range(max_leadtime):
#                 leadtime = deploy_durations[deploy_failures] - alarm_times[deploy_failures,L]
#                 pt.plot([L+1]*len(leadtime), leadtime, 'k.')
#             pt.xlim([-1, max_leadtime+1])
#             pt.xlabel("L")
#             pt.ylabel("Leadtime before failures")

#             pt.tight_layout()
#             pt.show()

# if __name__ == "__main__": main()


