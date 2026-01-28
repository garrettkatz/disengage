import itertools as it
import pickle as pk
import numpy as np
import torch as tr
from scipy.stats import beta
import matplotlib.pyplot as pt

def clopper_pearson(alpha, n, k):
    # Confidence intervals for binomial proportions
    # k successes out of n trials at alpha confidence level
    # expects k to be an array, lo/hi bounds are arrays of the same size
    # based on https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    p_lo = beta.ppf(alpha / 2, k, n - k + 1)
    p_hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    p_lo = np.where(np.isnan(p_lo), 0, p_lo)
    p_hi = np.where(np.isnan(p_hi), 1, p_hi)
    return p_lo, p_hi

def conform_experiments(params, obs_noises, delta_ratios, num_train_reps, setup_mu, EpisodeRunner):

    # unpack params
    basename = params["basename"]
    train_obs_noise = params["train_obs_noise"]
    max_episode_length = params["max_episode_length"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    calib_padding = params["calib_padding"]
    num_repetitions = params["num_repetitions"]

    # aggregate failure rate over all rollout buffers
    all_failures = []
    for rep in range(num_train_reps):
        buffer_name = f"{basename}_buffer_rep{rep}_on{train_obs_noise}.pt"
        (_, failures, _) = tr.load(buffer_name, weights_only=True)
        all_failures.append(failures)
    failrate = tr.cat(all_failures).mean().item()
    print(f"Overall failrate = {failrate}")

    # setup deltas and calibration counts
    deltas = [dr * failrate for dr in delta_ratios]
    num_calibration = int(np.ceil((1 - min(deltas) + calib_padding) / min(deltas))) # ensure non-infinite taus (unless no failures)
    print(f"{num_calibration} calibration samples based on deltas = {deltas}")

    # for each distribution shift
    for (calib_obs_noise, final_obs_noise) in it.combinations_with_replacement(obs_noises, 2):

        # loop experimental repetitions of conformal process
        for rep in range(num_repetitions):

            print(f"\n\n{calib_obs_noise=}, {final_obs_noise=}, {rep=}, collecting {num_calibration}+1 episodes...\n\n")

            # sample calibration episodes
            run_calib = EpisodeRunner(calib_obs_noise, max_episode_length)
            calib_observations, calib_failure = [], []
            for episode in range(num_calibration):
                failure, _, _, observations, _ = run_calib()
                calib_observations.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))
                calib_failure.append(failure)
            run_calib.close()
    
            run_final = EpisodeRunner(final_obs_noise, max_episode_length)
            final_failure, _, _, final_observations, _ = run_final()
            final_observations = tr.tensor(np.concatenate(final_observations, axis=0)).to(tr.float32)
            final_duration = len(final_observations)
            run_final.close()
            print(f"\n{final_failure=}, {final_duration=}\n")

            # initialize result storage
            results_name = f"{basename}_conform_rep{rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
                f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}.pkl"

            taus = np.empty((num_train_reps, 2, len(deltas), max_leadtime))
            alarm_times = np.empty((num_train_reps, 2, len(deltas), max_leadtime), dtype=int) # alarm_time == episode length signifies no alarm

            # process each train rep for later aggregation
            for train_rep in range(num_train_reps):

                # load pretrained state dicts
                train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_trained.pt"
                (_, _, _, mu_state_dicts) = tr.load(train_name, weights_only=True)

                # do once without and then with pretrained weights
                for load_mus in (False, True):

                    # setup randomly initialized mus
                    mu = {L: setup_mu() for L in range(1, max_leadtime + 1)}

                    # overwrite with trained weights when ready
                    if load_mus:
                        for L in mu: mu[L].load_state_dict(mu_state_dicts[L])

                    # conformal method for each L and delta
                    for L in range(1, max_leadtime + 1):

                        # calibration stats
                        calib_predictions = [mu[L](obs).squeeze() for obs in calib_observations]
                        z = [(preds[:max(1,len(preds)-L)].max().item() if fail else np.inf) for (preds, fail) in zip(calib_predictions, calib_failure)]
                        z.sort(reverse=True) # getting kth smallest, not largest

                        # predictions on final episode
                        final_preds = mu[L](final_observations).squeeze()

                        # test different deltas
                        for d, delta in enumerate(deltas):

                            # set tau
                            tau = z[int(np.ceil((num_calibration + 1)*(1 - delta)))-1] # -1 for 0-based indexing
                            print(f" {L=}, {delta=}, {tau=}")

                            # check for alarm on final episode
                            alarms = (final_preds[:max(1, len(final_preds)-L)] >= tau).numpy()

                            # save results
                            taus[train_rep, int(load_mus), d, L-1] = tau
                            if alarms.any():
                                alarm_times[train_rep, int(load_mus), d, L-1] = alarms.argmax()
                            else:
                                alarm_times[train_rep, int(load_mus), d, L-1] = final_duration

                    print(f"{train_rep=}, {load_mus=}, {final_failure=}, {final_duration=}, alarm times:")
                    print(alarm_times[train_rep, int(load_mus)])

            # save results
            with open(results_name, "wb") as f:
                pk.dump((failrate, delta_ratios, deltas, num_calibration, final_failure, final_duration, taus, alarm_times), f)


def show_results(params, obs_noises, delta_ratios, num_train_reps):

    pt.rcParams['font.family'] = 'serif'
    pt.rcParams['font.size'] = 12
    pt.rcParams['pdf.fonttype'] = 42

    basename = params["basename"]
    train_obs_noise = params["train_obs_noise"]
    num_repetitions = params["num_repetitions"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]

    pt.figure(figsize=(15,5))

    # want to show: some robustness of preemption rate under distribution shift
    for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

        Ls = np.arange(max_leadtime)+1
        unpreempted = np.empty((num_repetitions, num_train_reps, 2, len(delta_ratios), max_leadtime))

        # load results across each repetition
        for rep in range(num_repetitions):

            results_name = f"{basename}_conform_rep{rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
                f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}.pkl"
            with open(results_name, "rb") as f:
                (failrate, delta_ratios, deltas, num_calibration, final_failure, final_duration, taus, alarm_times) = pk.load(f)

            # unpreempted if failed and first alarm came at or later than max(1, fail time - L)
            unpreempted[rep] = final_failure & (alarm_times >= np.maximum(1, final_duration - Ls))

        # # unpreempted if failed and first alarm came at or later than max(1, fail time - L)
        # unpreempted[int(load_mus), train_rep, :, :, :] = deploy_failures[:,None,None] & (alarm_times >= np.maximum(1, deploy_durations[:,None,None] - Ls))

        # rates are average across conformal reps
        rand_rate = unpreempted[:,:,0,0,:].mean(axis=0)
        pret_rate = unpreempted[:,:,1,0,:].mean(axis=0)
        # mean/stdevs across training reps
        rand_mean = rand_rate.mean(axis=0)
        rand_stdv = rand_rate.std(axis=0)
        pret_mean = pret_rate.mean(axis=0)
        pret_stdv = pret_rate.std(axis=0)
        # confidence intervals over conformal reps
        rand_lo, rand_hi = clopper_pearson(.95, n=num_repetitions, k=unpreempted[:,:,0,0,:].sum(axis=0))
        pret_lo, pret_hi = clopper_pearson(.95, n=num_repetitions, k=unpreempted[:,:,1,0,:].sum(axis=0))
        # plot results
        pt.subplot(2,6,sp+1)
        pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
        pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
        # pt.fill_between(Ls, rand_lo.min(axis=0), rand_hi.max(axis=0), color=(1,.5,.5), alpha=.5)
        # pt.fill_between(Ls, pret_lo.min(axis=0), pret_hi.max(axis=0), color=(.5,.5,1), alpha=.5)
        # pt.plot(Ls, rand_rate.T, '-', color=(1,.7,.7))
        # pt.plot(Ls, pret_rate.T, '-', color=(.7,.7,1))
        pt.plot(Ls, rand_mean, 'ro:', label="random")
        pt.plot(Ls, pret_mean, 'b^-', label="trained")
        pt.plot(Ls, [deltas[0]]*max_leadtime, 'k--', label="delta")

        # pt.title(f"$\\nu_{{calib}}={calib_obs_noise}, \\nu_{{deploy}}={final_obs_noise}$")
        pt.title(f"({calib_obs_noise}, {final_obs_noise})")
        pt.xticks([],[])

        if sp==0: pt.ylabel("Unpreempted Failure")
    # pt.legend()
    # pt.gcf().supxlabel("L")
    # pt.gcf().supylabel("Unpreempted Failure Rate")
    # pt.tight_layout()
    # pt.savefig(f"{basename}_uf.pdf")
    # pt.show()

    # pt.figure(figsize=(15,3))

    # # want to show: average false alarm rates with trained mu are lower than untrained
    for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

        Ls = np.arange(max_leadtime)+1
        false_alarms = np.empty((num_repetitions, num_train_reps, 2, len(delta_ratios), max_leadtime))

        # load results across each repetition
        for rep in range(num_repetitions):

            results_name = f"{basename}_conform_rep{rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
                f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}.pkl"
            with open(results_name, "rb") as f:
                (failrate, delta_ratios, deltas, num_calibration, final_failure, final_duration, taus, alarm_times) = pk.load(f)

            # false alarm if not failed and first alarm came at any time before episode end
            false_alarms[rep] = (not final_failure) & (alarm_times < final_duration)

        # rates are average across conformal reps
        rand_rate = false_alarms[:,:,0,0,:].mean(axis=0)
        pret_rate = false_alarms[:,:,1,0,:].mean(axis=0)
        # mean/stdevs across training reps
        rand_mean = rand_rate.mean(axis=0)
        rand_stdv = rand_rate.std(axis=0)
        pret_mean = pret_rate.mean(axis=0)
        pret_stdv = pret_rate.std(axis=0)
        # confidence intervals over training reps
        rand_lo, rand_hi = clopper_pearson(.95, n=num_repetitions, k=false_alarms[:,:,0,0,:].sum(axis=0))
        pret_lo, pret_hi = clopper_pearson(.95, n=num_repetitions, k=false_alarms[:,:,1,0,:].sum(axis=0))

        # plot results
        pt.subplot(2,6,6+sp+1)
        pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
        pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
        # pt.fill_between(Ls, rand_lo.min(axis=0), rand_hi.max(axis=0), color=(1,.5,.5), alpha=.5)
        # pt.fill_between(Ls, pret_lo.min(axis=0), pret_hi.max(axis=0), color=(.5,.5,1), alpha=.5)
        # pt.plot(Ls, rand_rate.T, '-', color=(1,.7,.7))
        # pt.plot(Ls, pret_rate.T, '-', color=(.7,.7,1))
        pt.plot(Ls, rand_mean, 'ro:', label="random")
        pt.plot(Ls, pret_mean, 'b^-', label="trained")

        pt.subplot(2,6,6+sp+1)
        # # rates are average across conformal reps, then mean/std across training reps
        # rand_mean = false_alarms[:,:,0,0,:].mean(axis=0).mean(axis=0)
        # rand_stdv = false_alarms[:,:,0,0,:].mean(axis=0).std(axis=0)
        # pret_mean = false_alarms[:,:,1,0,:].mean(axis=0).mean(axis=0)
        # pret_stdv = false_alarms[:,:,1,0,:].mean(axis=0).std(axis=0)
        # pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
        # pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
        # pt.plot(Ls, rand_mean, 'ro:', label="random")
        # pt.plot(Ls, pret_mean, 'b^-', label="trained")
        # # pt.plot(Ls, [deltas[0]]*max_leadtime, 'k--', label="delta")
        # # pt.title(f"calib={calib_obs_noise}, deploy={final_obs_noise}")

        if sp==0: pt.ylabel("False Alarm")

    pt.legend()
    pt.gcf().supxlabel("L")
    # pt.gcf().supylabel("False Alarm Rate")
    pt.gcf().supylabel("Rates")
    pt.gcf().suptitle(f"$(\\nu_{{calib}}, \\nu_{{deploy}})$")
    pt.tight_layout()
    # pt.savefig(f"{basename}_fa.pdf")
    pt.savefig(f"{basename}_uf_fa.pdf")
    pt.show()

    # for sp, (calib_obs_noise, final_obs_noise) in enumerate(it.combinations_with_replacement(obs_noises, 2)):

    #     false_alarms = np.empty((2, 5, num_repetitions, len(delta_ratios), max_leadtime))
    #     for (load_mus, train_rep) in it.product((False, True), range(5)):

    #         basename = params["basename"]
    #         train_obs_noise = params["train_obs_noise"]
    #         # calib_obs_noise = params["calib_obs_noise"]
    #         # final_obs_noise = params["final_obs_noise"]
    #         max_leadtime = max_leadtime
    #         num_updates = params["num_updates"]
    #         learning_rate = params["learning_rate"]
    #         weight_decay = params["weight_decay"]
    #         results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
    #             f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

    #         # load preemption results
    #         with open(results_name, "rb") as f:
    #             (failrate, _, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)
        
    #         false_alarms[int(load_mus), train_rep, :, :, :] = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])

    #     pt.subplot(1,6,sp+1)
    #     rand_mean = false_alarms[0,:,:,0,:].mean(axis=1).mean(axis=0)
    #     rand_stdv = false_alarms[0,:,:,0,:].mean(axis=1).std(axis=0)
    #     pret_mean = false_alarms[1,:,:,0,:].mean(axis=1).mean(axis=0)
    #     pret_stdv = false_alarms[1,:,:,0,:].mean(axis=1).std(axis=0)
    #     Ls = np.arange(len(rand_mean))+1
    #     pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
    #     pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
    #     pt.plot(Ls, rand_mean, 'ro:', label="random")
    #     pt.plot(Ls, pret_mean, 'b^-', label="pretrained")
    #     # pt.xlabel("L")
    #     # pt.ylabel("False Alarm Rate")
    #     pt.title(f"calib={calib_obs_noise}, deploy={final_obs_noise}")

    # pt.legend()
    # pt.gcf().supxlabel("L")
    # pt.gcf().supylabel("False Alarm Rate")
    # pt.tight_layout()
    # pt.savefig(f"{basename}_fa.pdf")
    # pt.show()


# ### save below until:
# [ ] lunar lander
# [ ] cartpole
# [ ] humanoid (train_obs_noise only)

# def conform(params, delta_ratios, setup_mu, run_calib, run_final):

#     basename = params["basename"]
#     train_rep = params["train_rep"]
#     train_obs_noise = params["train_obs_noise"]
#     calib_obs_noise = params["calib_obs_noise"]
#     final_obs_noise = params["final_obs_noise"]
#     max_leadtime = params["max_leadtime"]
#     num_updates = params["num_updates"]
#     learning_rate = params["learning_rate"]
#     weight_decay = params["weight_decay"]
#     calib_padding = params["calib_padding"]
#     num_repetitions = params["num_repetitions"]
#     load_mus = params["load_mus"]

#     train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_trained.pt"
#     results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
#         f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

#     # load failrate and mus
#     (failrate, _, _, mu_state_dicts) = tr.load(train_name, weights_only=True)
#     mu = {}
#     for L in range(1, max_leadtime + 1):
#         mu[L] = setup_mu()
#         if load_mus: mu[L].load_state_dict(mu_state_dicts[L])

#     # setup calibration parameters
#     deltas = [dr * failrate for dr in delta_ratios]
#     num_calibration = int(np.ceil((1 - min(deltas) + calib_padding) / min(deltas))) # ensure non-infinite taus (unless no failures)

#     deploy_failures = np.empty(num_repetitions, dtype=bool)
#     deploy_durations = np.empty(num_repetitions, dtype=int)
#     taus = np.empty((num_repetitions, len(deltas), max_leadtime))
#     alarm_times = np.empty((num_repetitions, len(deltas), max_leadtime), dtype=int) # episode length signifies no alarm
#     for rep in range(num_repetitions):

#         # collect calibration episodes
#         print(f"collecting {num_calibration} episodes...")
#         calib_observations, calib_failure = [], []
#         for episode in range(num_calibration):
#             failure, _, _, observations, _ = run_calib()
#             calib_observations.append(tr.tensor(np.concatenate(observations, axis=0)).to(tr.float32))
#             calib_failure.append(failure)

#         # final episode
#         final_failure, _, _, final_observations, _ = run_final()
#         final_observations = tr.tensor(np.concatenate(final_observations, axis=0)).to(tr.float32)
#         deploy_failures[rep] = final_failure
#         deploy_durations[rep] = len(final_observations)

#         # conformal method for each L and delta
#         for L in range(1, max_leadtime + 1):

#             # calibration stats
#             calib_predictions = [mu[L](obs).squeeze() for obs in calib_observations]
#             z = [(preds[:max(1,len(preds)-L)].max().item() if fail else np.inf) for (preds, fail) in zip(calib_predictions, calib_failure)]
#             z.sort(reverse=True) # getting kth smallest, not largest

#             # predictions on final episode
#             final_preds = mu[L](final_observations).squeeze()
            
#             # test different deltas
#             for d, delta in enumerate(deltas):

#                 # set tau
#                 tau = z[int(np.ceil((num_calibration + 1)*(1 - delta)))-1] # -1 for 0-based indexing
#                 print(f" {L=}, {delta=}, {tau=}")
    
#                 # check for alarm on final episode
#                 alarms = (final_preds[:max(1, len(final_preds)-L)] >= tau).numpy()

#                 # save results
#                 taus[rep, d, L-1] = tau
#                 if alarms.any():
#                     alarm_times[rep, d, L-1] = alarms.argmax()
#                 else:
#                     alarm_times[rep, d, L-1] = len(final_observations)

#         for d, delta in enumerate(deltas):
#             print(f" {failrate=}, {num_updates=}, {delta=}, {num_calibration=}, {rep=}, {final_failure=}, dur={len(final_observations)}, alarms at:", alarm_times[rep, d])

#     print(f"\n\n **** {load_mus=}, {train_rep=}, {calib_obs_noise=}, {final_obs_noise=} ***\n\n")

#     print(f"{failrate=}, {num_updates=}, {num_calibration=}, {deltas=}:")
#     sound_alarms =  deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
#     false_alarms = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
#     unpreempted = deploy_failures[:,None,None] & (alarm_times==deploy_durations[:,None,None])
#     print(f" unpreempted rates (delta x L):")
#     print(unpreempted.mean(axis=0).round(3))
#     print(f" false alarm rates (delta x L):")
#     print(false_alarms.mean(axis=0).round(3))

#     with open(results_name, "wb") as f:
#         pk.dump((failrate, delta_ratios, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times), f)


# def show_results(params):

#     basename = params["basename"]
#     train_rep = params["train_rep"]
#     train_obs_noise = params["train_obs_noise"]
#     calib_obs_noise = params["calib_obs_noise"]
#     final_obs_noise = params["final_obs_noise"]
#     max_leadtime = params["max_leadtime"]
#     num_updates = params["num_updates"]
#     learning_rate = params["learning_rate"]
#     weight_decay = params["weight_decay"]
#     load_mus = params["load_mus"]

#     train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_trained.pt"
#     results_name = f"{basename}_conform_rep{train_rep}_tn{train_obs_noise}_cn{calib_obs_noise}_fn{final_obs_noise}_" +\
#         f"lr{learning_rate}_wd{weight_decay}_lt{max_leadtime}_nu{num_updates}_lm{load_mus}.pkl"

#     # load preemption results
#     with open(results_name, "rb") as f:
#         (failrate, delta_ratios, deltas, num_calibration, deploy_failures, deploy_durations, alarm_times) = pk.load(f)

#     print(f"{failrate=}, {num_updates=}, {num_calibration=}, {deltas=}:")
#     sound_alarms =  deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
#     false_alarms = ~deploy_failures[:,None,None] & (alarm_times < deploy_durations[:,None,None])
#     unpreempted = deploy_failures[:,None,None] & (alarm_times==deploy_durations[:,None,None])
#     print(f" unpreempted rates (delta x L):")
#     print(unpreempted.mean(axis=0).round(3))
#     print(f" false alarm rates (delta x L):")
#     print(false_alarms.mean(axis=0).round(3))

#     # pt.subplot(1,3,1)
#     for d, (dr, delta) in enumerate(zip(delta_ratios, deltas)):
#         pt.plot(np.arange(1, max_leadtime+1), false_alarms[:,d,:].mean(axis=0), 'x-', color=(dr,dr,1), label=f"false alarms (delta={delta})")
#         pt.plot(np.arange(1, max_leadtime+1), unpreempted[:,d,:].mean(axis=0), 'o-', color=(1,dr,dr), label=f"unpreempted failures (delta={delta})")
#         pt.plot(np.arange(1, max_leadtime+1), [delta]*max_leadtime, ':', color=(dr,)*3, label=f"delta={delta}")
#     pt.xlabel("L")
#     pt.ylabel("Rate")
#     pt.legend()

#     # pt.subplot(1,3,2)
#     # for L in range(max_leadtime):
#     #     quiet_time = alarm_times[~deploy_failures,L] / deploy_durations[~deploy_failures]
#     #     pt.plot([L+1]*len(quiet_time), quiet_time, 'k.')
#     # pt.xlim([-1, max_leadtime+1])
#     # pt.xlabel("L")
#     # pt.ylabel("Alarm-free portion of non-failures")

#     # pt.subplot(1,3,3)
#     # for L in range(max_leadtime):
#     #     leadtime = deploy_durations[deploy_failures] - alarm_times[deploy_failures,L]
#     #     pt.plot([L+1]*len(leadtime), leadtime, 'k.')
#     # pt.xlim([-1, max_leadtime+1])
#     # pt.xlabel("L")
#     # pt.ylabel("Leadtime before failures")

#     # pt.tight_layout()
#     pt.show()

