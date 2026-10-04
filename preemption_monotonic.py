import itertools as it
import pickle as pk
import numpy as np
import torch as tr
from scipy.stats import beta
import matplotlib.pyplot as pt

def conform_experiments(params, obs_noises, delta_ratios, num_train_reps, setup_mu, EpisodeRunner):

    # unpack params
    basename = params["basename"]
    buf_basename = params["buf_basename"]
    train_obs_noise = params["train_obs_noise"]
    max_episode_length = params["max_episode_length"]
    max_leadtime = params["max_leadtime"]
    num_updates = params["num_updates"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    calib_padding = params["calib_padding"]
    num_repetitions = params["num_repetitions"]

    Ls = list(range(1, max_leadtime + 1))

    # aggregate failure rate over all rollout buffers
    all_failures = []
    for rep in range(num_train_reps):
        buffer_name = f"{buf_basename}_buffer_rep{rep}_on{train_obs_noise}.pt"
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
                train_name = f"{basename}_train_rep{train_rep}_on{train_obs_noise}_lr{learning_rate}_wd{weight_decay}_nu{num_updates}_trained.pt"
                (_, _, _, mu_state_dict) = tr.load(train_name, weights_only=True)

                # do once without and then with pretrained weights
                for load_mus in (False, True):

                    # setup randomly initialized mu
                    mu = setup_mu()

                    # overwrite with trained weights when ready
                    if load_mus:
                        mu.load_state_dict(mu_state_dict)

                    # conformal method for each L and delta
                    for L in Ls:

                        # calculate per-timestep ratios on each calib episode
                        ratios = []
                        for obs, fff in zip(calib_observations, calib_failure):
                            all_probs = tr.sigmoid(mu(obs[:-1])) # (ep len-1, horizons)
                            numers = all_probs[...,L-2] if L > 1 else 1. # [t+1,t+L) = [t+1,t+L-1] and -1 more for horizon indexing
                            denoms = tr.take_along_dim(all_probs, max_episode_length-1 - tr.arange(len(obs)-1)[:,None] -1, dim=-1).squeeze()
                            ratios.append( numers / denoms )
                            # if load_mus:
                            #     print(f"{len(obs)} timesteps, failure={fff}")
                            #     print('numer', numers)
                            #     print('denom', denoms)
                            #     print('ratio', ratios[-1])
                            #     input('.')

                        # calibration stats
                        z = [(preds[:max(1,len(preds)-L)].max().item() if fail else np.inf) for (preds, fail) in zip(ratios, calib_failure)]
                        z.sort(reverse=True) # getting kth smallest, not largest

                        # predictions on final episode
                        obs = final_observations
                        all_probs = tr.sigmoid(mu(obs[:-1])) # (ep len-1, horizons)
                        numers = all_probs[...,L-2] if L > 1 else 1. # [t+1,t+L) = [t+1,t+L-1] and -1 more for horizon indexing
                        denoms = tr.take_along_dim(all_probs, max_episode_length-2 - tr.arange(len(obs)-1)[:,None], dim=-1).squeeze()
                        final_preds = numers / denoms

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
        # plot results
        pt.subplot(2,6,sp+1)
        pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
        pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
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
        # print('pre-rep pret_fa[5], mean:')
        # print(pret_rate[:,5-1]) # -1 since L=1 starts at index 0
        # print(pret_rate[:,5-1].mean())
        # input('.')

        # plot results
        pt.subplot(2,6,6+sp+1)
        pt.fill_between(Ls, rand_mean-rand_stdv, rand_mean+rand_stdv, color=(1,.5,.5), alpha=.5)
        pt.fill_between(Ls, pret_mean-pret_stdv, pret_mean+pret_stdv, color=(.5,.5,1), alpha=.5)
        pt.plot(Ls, rand_mean, 'ro:', label="random")
        pt.plot(Ls, pret_mean, 'b^-', label="trained")

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


if __name__ == "__main__":

    from time import perf_counter
    from train_mu_monotonic import setup_mu_factory
    from ruins_ratio import EpisodeRunner

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
        "learning_rate": 1e-5,
        "weight_decay": .1,

        "calib_padding": 0, # this many outliers below tau
        "num_repetitions": 100, # this many repetitions to estimate preemption rates

    }

    setup_mu = setup_mu_factory(params["max_episode_length"])

    start_time = perf_counter()
    conform_experiments(params, obs_noises, delta_ratios, num_train_reps, setup_mu, EpisodeRunner)
    print(f"Experiments took {perf_counter() - start_time}s")

    show_results(params, obs_noises, delta_ratios, num_train_reps)
